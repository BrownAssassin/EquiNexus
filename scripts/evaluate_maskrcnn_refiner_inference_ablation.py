# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

"""Evaluate guarded refiner TTA and fallback policies on the frozen holdout."""
from __future__ import annotations
import csv, hashlib, json, sys, time
from pathlib import Path
import cv2, numpy as np, torch
from pycocotools import mask as mu
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
import filament_baseline.external_panoptic as ep
from filament_baseline.data import load_dataset
from filament_baseline.predictor import PredictedInstance, read_grayscale
from filament_baseline.runtime import hardware_info, package_versions
from filament_baseline.scorer import score_validation
from filament_baseline.split import create_observation_split
from filament_baseline.visuals import write_comparison_overlays
from run_maskrcnn_experiment import merge_positive_overlaps
from run_maskrcnn_gt_crop_refiner import _square_crop, encode_refined_predictions, score_refined_epoch, load_detection_payload
from score_detector_json import score_detector_predictions
from evaluate_component_selector_probability_blend_holdout import paired_bootstrap, per_observation_rows

DATA=ROOT/'data/raw/MAGFiLO_1.0_Kaggle_2026'
DETECTOR=ROOT/'artifacts/detector-experiments/maskrcnn-r50fpn-v2-1024-pq-holdout-20261110-8ep'
REFINER=ROOT/'artifacts/detector-experiments/maskrcnn-r50fpn-v2-gt-crop-refiner-256-holdout-20261110'
OUT=ROOT/'artifacts/detector-experiments/maskrcnn-refiner-inference-ablation-repro-20261110'
OUT.mkdir(parents=True,exist_ok=True)
started=time.perf_counter(); device=torch.device('cuda:0'); torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
ds=load_dataset(DATA); split=create_observation_split(ds,ROOT/'data/splits'); names=tuple(sorted(split.validation_names)); stems={Path(n).stem for n in names}
expected='30cb2ec52437daf97e769a3a61bf83350062e6cf4f5d00af1f8399649065e48e'
assert split.sha256==expected
raw=load_detection_payload(DETECTOR/'best-validation-detections.json',names)
ckpt=torch.load(REFINER/'best.pt',map_location=device,weights_only=False)
model=ep.CropRefiner().to(device); model.load_state_dict(ckpt['model'],strict=True); model.eval()
# The crop-refiner checkpoint's selected threshold was 0.50; use it in the same topology guard.
ep.REFINER_THRESHOLD=0.50
modes={name:{stem:[] for stem in stems} for name in ('raw_single_drop','raw_single_fallback','raw_tta_drop','raw_tta_fallback','raw_safe_tta_drop','raw_safe_tta_fallback','premerge_safe_tta_fallback')}
empty_refinements=0; total_refinements=0; premerge_counts={'before':0,'after':0}
raw_stage=OUT/'raw-stage.json'

def add_crops(source, mode_prefix):
    global empty_refinements,total_refinements
    crops=[]; records=[]
    for file_name in names:
        stem=Path(file_name).stem
        image=read_grayscale(ds.train_image_dir/file_name)
        items=source.get(stem,[])
        for item in items:
            score=float(item['score'])
            if score<0.60: continue
            fallback_rle={'size':[2048,2048],'counts':item['counts'].encode('ascii')}
            fallback=mu.decode(fallback_rle).astype(bool)
            if not fallback.any(): continue
            x0,y0,x1,y1=_square_crop(fallback,context=1.8,minimum=96)
            crop=cv2.resize(image[y0:y1,x0:x1],(256,256),interpolation=cv2.INTER_LINEAR)
            crops.append(crop); records.append((stem,score,(x0,y0,x1,y1),fallback_rle))
    if not crops: raise RuntimeError('No crops generated')
    crops=np.stack(crops).astype(np.uint8,copy=False)
    p0=ep._refiner_probabilities(model,crops,device,batch_size=24)
    flip_probs=[]
    for transform in ('horizontal','vertical','both'):
        values=ep._refiner_probabilities(model,ep._transform_batch(crops,transform),device,batch_size=24)
        flip_probs.append(ep._undo_transform(values,transform).astype(np.float32,copy=False))
    ptta=np.mean(np.stack([p0,*flip_probs]),axis=0,dtype=np.float32)
    def render(prob, fallback_rle, bounds, use_fallback):
        x0,y0,x1,y1=bounds
        native=cv2.resize(prob,(x1-x0,y1-y0),interpolation=cv2.INTER_LINEAR)
        binary=native>=0.50
        if not binary.any():
            return fallback_rle if use_fallback else None
        from filament_baseline.predictor import encode_cropped_binary_mask
        return {'size':[2048,2048],'counts':encode_cropped_binary_mask(binary,x0,y0,(2048,2048))['counts']}
    for index,(stem,score,bounds,fallback_rle) in enumerate(records):
        total_refinements+=1
        if not np.any(cv2.resize(p0[index],(bounds[2]-bounds[0],bounds[3]-bounds[1]),interpolation=cv2.INTER_LINEAR)>=0.50): empty_refinements+=1
        probs={'raw_single':p0[index],'raw_tta':ptta[index]}
        safe=ep._safe_tta_mask(p0[index],ptta[index])
        probs['raw_safe_tta']=safe
        for name,prob in probs.items():
            mask=render(prob,fallback_rle,bounds,False)
            if mask is not None: modes[name+'_drop'][stem].append({'score':score,'counts':mask['counts'] if isinstance(mask['counts'],str) else mask['counts'].decode('ascii')})
            mask_fb=render(prob,fallback_rle,bounds,True)
            if mask_fb is not None: modes[name+'_fallback'][stem].append({'score':score,'counts':mask_fb['counts'] if isinstance(mask_fb['counts'],str) else mask_fb['counts'].decode('ascii')})
    return len(records)

if raw_stage.is_file():
    cached=json.loads(raw_stage.read_text(encoding='utf-8'))
    for key in ('raw_single_drop','raw_single_fallback','raw_tta_drop','raw_tta_fallback','raw_safe_tta_drop','raw_safe_tta_fallback'):
        modes[key]=cached[key]
else:
    raw_n=add_crops(raw,'raw')
    cached={key:modes[key] for key in ('raw_single_drop','raw_single_fallback','raw_tta_drop','raw_tta_fallback','raw_safe_tta_drop','raw_safe_tta_fallback')}
    raw_stage.write_text(json.dumps(cached,separators=(',',':')),encoding='utf-8')
merged={}
for stem,items in raw.items():
    eligible=[item for item in items if float(item['score'])>=0.60]
    premerge_counts['before']+=len(eligible)
    merged[stem]=merge_positive_overlaps(eligible)
    premerge_counts['after']+=len(merged[stem])
# The public reference's detector pre-merge + guarded TTA + fallback pipeline, with our own checkpoints.
# Recompute the actual pre-merged variant separately, since the crops and fallback masks differ.
def evaluate_source(source,key):
    crops=[]; records=[]
    for file_name in names:
        stem=Path(file_name).stem; image=read_grayscale(ds.train_image_dir/file_name)
        for item in source.get(stem,[]):
            score=float(item['score'])
            if score<0.60: continue
            rle={'size':[2048,2048],'counts':item['counts'].encode('ascii')}; fallback=mu.decode(rle).astype(bool)
            if not fallback.any(): continue
            bounds=_square_crop(fallback,context=1.8,minimum=96); x0,y0,x1,y1=bounds
            crops.append(cv2.resize(image[y0:y1,x0:x1],(256,256),interpolation=cv2.INTER_LINEAR))
            records.append((stem,score,bounds,rle))
    array=np.stack(crops).astype(np.uint8,copy=False)
    p0=ep._refiner_probabilities(model,array,device,batch_size=24); pp=[p0]
    for transform in ('horizontal','vertical','both'):
        q=ep._refiner_probabilities(model,ep._transform_batch(array,transform),device,batch_size=24)
        pp.append(ep._undo_transform(q,transform).astype(np.float32,copy=False))
    mean=np.mean(np.stack(pp),axis=0,dtype=np.float32); pred={stem:[] for stem in stems}
    from filament_baseline.predictor import encode_cropped_binary_mask
    for (stem,score,bounds,fallback),original,average in zip(records,p0,mean,strict=True):
        prob=ep._safe_tta_mask(original,average); x0,y0,x1,y1=bounds
        native=cv2.resize(prob,(x1-x0,y1-y0),interpolation=cv2.INTER_LINEAR); binary=native>=0.50
        if binary.any():
            value=encode_cropped_binary_mask(binary,x0,y0,(2048,2048))['counts']
            counts=value if isinstance(value,str) else value.decode('ascii')
        else: counts=fallback['counts'].decode('ascii')
        pred[stem].append({'score':score,'counts':counts})
    modes[key]=pred

evaluate_source(merged,'premerge_safe_tta_fallback')
results=[]; rows_by_mode={}
for key,pred in modes.items():
    metrics,rows,count=score_detector_predictions(ds,names,pred,confidence_threshold=0.60,minimum_area=200)
    rows_by_mode[key]=rows
    results.append({'mode':key,'global':metrics['global'],'prediction_count':count,'site':metrics['by_site'],'year':metrics['by_year']})
    outpath=OUT/f'{key}.json'; outpath.write_text(json.dumps(pred,separators=(',',':')),encoding='utf-8')
    print(f"{key}: PQ={metrics['global']['pq']:.6f} TP={metrics['global']['tp']} FP={metrics['global']['fp']} FN={metrics['global']['fn']} predictions={count}",flush=True)
base_rows=rows_by_mode['raw_single_drop']; best_key=max(results,key=lambda x:x['global']['pq'])['mode']; best_rows=rows_by_mode[best_key]
report={'status':'complete','source':'public Apache-2.0 inference ideas (four-flip refiner TTA, topology guard, and empty-mask fallback) evaluated with EquiNexus-owned checkpoints','split_sha256':split.sha256,'validation_observations':len(names),'validation_annotator_views':sum(len(ds.views_by_observation[n]) for n in names),'test_images_or_labels_used':False,'detector_score_threshold':0.60,'minimum_instance_area':200,'refiner_threshold':0.50,'refiner_checkpoint_sha256':hashlib.sha256((REFINER/'best.pt').read_bytes()).hexdigest(),'detector_prediction_sha256':hashlib.sha256((DETECTOR/'best-validation-detections.json').read_bytes()).hexdigest(),'model_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'empty_refiner_masks_at_threshold':empty_refinements,'total_refiner_masks':total_refinements,'premerge_prediction_counts':premerge_counts,'results':results,'best_mode':best_key,'paired_bootstrap_vs_raw_no_fallback':paired_bootstrap(base_rows,best_rows),'per_observation_best':per_observation_rows(best_rows),'hardware':hardware_info(),'packages':package_versions(),'peak_gpu_memory_bytes':int(torch.cuda.max_memory_allocated(device)),'elapsed_seconds':time.perf_counter()-started}
(OUT/'metrics.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps({'best_mode':best_key,'paired_bootstrap':report['paired_bootstrap_vs_raw_no_fallback'],'empty_refiner_masks':empty_refinements,'total':total_refinements,'runtime_seconds':report['elapsed_seconds'],'output':str(OUT)},indent=2))
