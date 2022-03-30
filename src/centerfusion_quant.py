#use command line options:
# ddd --load_model ../models/nuScenes_3Ddetection_e140.pth --gpus 0 --dataset nuscenes --dla_node conv --data_dir /media/ronny/dataset --quant_mode calib --batch_size 1 --num_workers 0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths

import os
import json
import pytorch_nndct
import torch
from pytorch_nndct.apis import torch_quantizer, dump_xmodel
import numpy as np

from opts import opts
from logger import Logger
from dataset.dataset_factory import dataset_factory
from detector_quant import DetectorQuant
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter
from nndct_shared.utils import NndctDebugLogger, NndctOption, PatternType, NndctScreenLogger
from progress.bar import Bar

NndctOption.nndct_parse_debug.value = 4

def do_trace(model, inp):
  model_trace = torch.jit.trace(model, inp, check_trace=False)
  model_trace = model_trace.eval()
  return model_trace


def dict_to_tuple(out_dict):
  return out_dict["hm"], out_dict["reg"], out_dict["wh"], out_dict["dep"], out_dict["rot"], out_dict["dim"], \
         out_dict["amodel_offset"], out_dict["nuscenes_att"], out_dict["velocity"]

def dict_to_tupleSecHead(out_dict):
  return out_dict["hm"], out_dict["reg"], out_dict["wh"], out_dict["dep"], out_dict["rot"], out_dict["dim"], \
         out_dict["amodel_offset"], out_dict["pc_hm"], out_dict["velocity"], \
         out_dict["nuscenes_att"], out_dict["dep_sec"], out_dict["rot_sec"]

#def dict_to_tuple(out_dict):
#  return out_dict["hm"], out_dict["reg"], out_dict["wh"], out_dict["dep"], out_dict["rot"], out_dict["dim"], \
#         out_dict["amodel_offset"], out_dict["nuscenes_att"]

class DLASegWrapper(torch.nn.Module):
  def __init__(self, model):
    super().__init__()
    self.model = model

  #def forward(self, x, pc_hm=None, pc_dep=None, calib=None):
  def forward(self, x):
    out = self.model(x, pc_hm=None, pc_dep=None, calib=None)
    return dict_to_tuple(out[0])

class DLASecHeeadWrapper(torch.nn.Module):
  def __init__(self, model):
    super().__init__()
    self.model = model

  def forward(self, x, pc_dep=None, calib=None):
    out = self.model(x, pc_hm=None, pc_dep=pc_dep, calib=calib)
    return dict_to_tupleSecHead(out[0])



def evaluate(detector, dataset, opt):
  num_iters = len(dataset) if opt.num_iters < 0 else opt.num_iters
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  Loss = 0
  total = 0
  dataset_iterator = iter(dataset)
  for ind in range(num_iters):
    img_tensor = next(dataset_iterator)
    loss_total, losses = detector.run(img_tensor)
    Loss += loss_total
    total += 1
    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} | tot_Loss: {totloss:}'.format(
      ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td, totloss=loss_total)
    bar.next()
  bar.finish()
  return Loss / total

def get_children(model: torch.nn.Module):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
      # if model has no children; model is last child! :O
      return model
    else:
      # look for children from children... to the last child!
      for child in children:
        try:
          flatt_children.extend(get_children(child))
        except TypeError:
          flatt_children.append(get_children(child))
    return flatt_children


if __name__ == '__main__':
  opt = opts().parse()
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  if opt.gpus[0] >= 0:
    opt.device = torch.device('cuda')
  else:
    opt.device = torch.device('cpu')

  if opt.deploy:
    opt.num_iters = 1

  NndctDebugLogger(os.path.join(opt.save_dir, "debug_Jit.txt"))

  Dataset = dataset_factory[opt.test_dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  Logger(opt)

  split = 'val' if not opt.trainval else 'test'
  if split == 'val':
    split = opt.val_split


  dataset = torch.utils.data.DataLoader(
        Dataset(opt, split), batch_size=opt.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True)

  detector = None

  if opt.quant_mode == 'float':
    detector = DetectorQuant(opt)
    detector.setquantmodel(detector.model)
  else:

    detector = DetectorQuant(opt)
    dataset_iterator = iter(dataset)
    img_tensor = next(dataset_iterator)
    #input = torch.cat((img_tensor['image'], img_tensor['image']))
    input = img_tensor['image']
    input = input.to(device=opt.device, non_blocking=True)
    #pc_dep = torch.cat((img_tensor['pc_dep'], img_tensor['pc_dep']))
    if opt.pointcloud:
      pc_dep = img_tensor['pc_dep']
      pc_dep = pc_dep.to(device=opt.device, non_blocking=True)
      #pc_hm = torch.cat((img_tensor['pc_hm'], img_tensor['pc_hm']))
      pc_hm = img_tensor['pc_hm']
      pc_hm = pc_hm.to(device=opt.device, non_blocking=True)

    calib = torch.randn([img_tensor['calib'].shape[1], img_tensor['calib'].shape[2]])
    calib = calib.to(device=opt.device, non_blocking=True)

    #modelout = detector.model(input, pc_hm, pc_dep, calib)
    #pytorch_nndct.NndctOption().nndct_parse_debug.value = 2
    #module_model = torch.jit.trace(detector.model, (input, pc_hm, pc_dep, calib))
    if opt.pointcloud:
      modelWrap = DLASecHeeadWrapper(detector.model)
    else:
      modelWrap = DLASegWrapper(detector.model)

    modelWrap.eval()

    #with torch.no_grad():
    #out = modelWrap(input, pc_hm, pc_dep, calib)
    #if opt.pointcloud:
    #  script_module = do_trace(modelWrap, (input, pc_hm, pc_dep, calib))
    #else:
    #  script_module = do_trace(modelWrap, (input))

    #savetraceModel = os.path.join(opt.save_dir, "traceModel.pt")
    #script_module.save(savetraceModel)
    #print(script_module.code)
    #writer = SummaryWriter(opt.save_dir)
    #writer.add_graph(modelWrap, input_to_model=(input, pc_hm, pc_dep, calib), verbose=True)
    #writer.close()
    ##quantizer = torch_quantizer(quant_mode=opt.quant_mode, module=modelWrap, input_args=(input, pc_hm, pc_dep, calib), output_dir=opt.save_dir)
    if opt.pointcloud:
      quantizer = torch_quantizer(quant_mode=opt.quant_mode, module=modelWrap, input_args=(input, pc_dep, calib),
                                output_dir=opt.save_dir)
    else:
      quantizer = torch_quantizer(quant_mode=opt.quant_mode, module=modelWrap, input_args=(input),
                                  output_dir=opt.save_dir)

    detector.setquantmodel(quantizer.quant_model)

  if opt.fast_finetune == True:
    if opt.quant_mode == 'calib':
        quantizer.fast_finetune(evaluate, (detector, dataset, opt))
    elif opt.quant_mode == 'test':
        quantizer.load_ft_param()
  loss_gen = evaluate(detector, dataset, opt)
  print('loss: %g' % (loss_gen))

  # handle quantization result
  if opt.quant_mode == 'calib':
    quantizer.export_quant_config()

  if opt.deploy:
    quantizer.export_xmodel(output_dir=opt.save_dir, deploy_check=False)



