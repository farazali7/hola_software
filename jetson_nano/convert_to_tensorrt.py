import torch
from torch2trt import torch2trt  # NEED TO DO THIS ON JETSON NANO ITSELF
from classification.src.models.models import get_model
from classification.src.config import cfg as cls_cfg
from jetson_nano import config as jn_cfg
from torchmetrics import MetricCollection, Precision, Recall, F1Score
from copy import deepcopy


# Load pretrained model state dict
model_path = jn_cfg['PT_MODEL_PATH']
model_pth = torch.load(model_path)

# Params and metrics for trainer (just used for backward compatability here)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = cls_cfg['CLASSES']
num_classes = len(classes)
metrics = MetricCollection({
    'Multiclass Recall': Recall(task='multiclass', num_classes=num_classes, average=None),
    'Multiclass Precision': Precision(task='multiclass', num_classes=num_classes, average=None),
    'Multiclass F1-Score': F1Score(task='multiclass', num_classes=num_classes, average=None),
})
class_weights = torch.Tensor([0.6, 0.9, 0.9]).to(device)

trainer_args = {'classes': classes,
                'metrics': metrics,
                'learning_rate': 0,
                'class_weights': class_weights,
                'prev_optimizer_state': deepcopy(model_pth['optimizer_states'][0])}

model_args = {'dropout': 0}

# Get base model architecture
model = get_model(model_name=cls_cfg['MODEL_ARCHITECTURE'], model_args=model_args, trainer_args=trainer_args,
                  use_legacy=False)
model.load_state_dict(model_pth['state_dict']).eval().cuda()

# create example data
x = torch.ones((1, 5, 5, 5)).cuda()

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x])


torch.save(model_trt.state_dict(), 'model_trt.pth')
