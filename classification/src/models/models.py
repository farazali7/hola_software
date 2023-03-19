import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl


def get_model(model_name, model_args, trainer_args):
    """
    Retrieve a module wrapped around specified model, initialized with proper arguments.
    :param model_name: String for underlying model architecture
    :param model_args: Dictionary of model architecture arguments
    :param trainer_args: Keyword arguments for model trainer such as learning rate, class weights, etc.
    :return: Model
    """
    if model_name == 'MLP':
        model_def = MLP_MODEL(model_args)
    elif model_name == 'MLP_ITER2':
        model_def = MLP_MODEL_ITER2(model_args)
    elif model_name == 'CNN':
        model_def = CNN_MODEL(model_args)
    else:
        raise Exception(f'Given model name: {model_name} is not supported.')

    model = Model(model_def, **trainer_args)

    return model


def load_model_from_checkpoint(checkpoint_path):
    """
    Load and return a model from a given checkpoint path.
    :param checkpoint_path: String for location of model
    :return: Model from checkpoint
    """
    model = Model.load_from_checkpoint(checkpoint_path=checkpoint_path)

    return model


class Model(pl.LightningModule):
    def __init__(self, model, learning_rate, class_weights, classes, metrics, fold=None):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.class_weights = class_weights
        self.classes = classes
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')
        self.fold = fold
        self.log_prefix = 'fold_'+str(self.fold)+'/'

        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x)
        targets = torch.squeeze(y)
        loss = nn.CrossEntropyLoss(weight=self.class_weights)(outputs, targets)

        _, preds = torch.max(outputs.detach(), 1)

        # Log and accumulate
        res = {'loss': loss, 'preds': preds, 'targets': targets}
        self.train_step_outputs.append(res)

        train_metrics_out = self.train_metrics(preds, targets)
        self.log_metrics(train_metrics_out)
        self.log(self.log_prefix+'train_loss', loss, prog_bar=True)

        return res

    def on_train_epoch_end(self):
        train_metrics_out = self.train_metrics.compute()
        self.log_metrics(train_metrics_out)
        self.train_metrics.reset()

        step_losses = [step_item['loss'] for step_item in self.train_step_outputs]
        epoch_loss = torch.stack(step_losses).mean()
        self.log(self.log_prefix+'epoch_train_loss', epoch_loss)
        self.train_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x)
        targets = torch.squeeze(y)
        val_loss = nn.CrossEntropyLoss()(outputs, targets)

        _, preds = torch.max(outputs.detach(), 1)

        # Log and accumulate
        res = {'val_loss': val_loss, 'preds': preds, 'targets': targets}
        self.val_step_outputs.append(res)

        self.val_metrics.update(preds, targets)
        self.log(self.log_prefix+'val_loss', val_loss, prog_bar=True)

        return res

    def on_validation_epoch_end(self):
        valid_metrics_out = self.val_metrics.compute()
        self.log_metrics(valid_metrics_out)
        self.val_metrics.reset()

        step_losses = [step_item['val_loss'] for step_item in self.val_step_outputs]
        epoch_loss = torch.stack(step_losses).mean()
        self.log(self.log_prefix+'epoch_val_loss', epoch_loss)
        self.val_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x)
        targets = torch.squeeze(y)
        test_loss = nn.CrossEntropyLoss()(outputs, targets)

        _, preds = torch.max(outputs.detach(), 1)

        # Log and accumulate
        res = {'test_loss': test_loss, 'preds': preds, 'targets': targets}
        self.test_step_outputs.append(res)
        self.test_metrics.update(preds, targets)

        return res

    def on_test_epoch_end(self):
        test_metrics_out = self.test_metrics.compute()
        self.log_metrics(test_metrics_out)
        self.test_metrics.reset()

        self.test_step_outputs.clear()

    def predict_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x)
        targets = torch.squeeze(y)

        _, preds = torch.max(outputs.detach(), 1)

        return {'preds': preds, 'targets': targets}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), self.learning_rate)

        return [optimizer]

    def log_metrics(self, metrics_val_dict):
        """
        Delineate when logging MetricCollections for scalar and tensor-returning metrics.
        :param metrics_val_dict: Dictionary of metric values to log
        :return: None, simply logs to logger
        """
        scalar_dict = {}
        for metric_name in metrics_val_dict:
            metric_tensor = metrics_val_dict[metric_name]
            new_name = self.log_prefix+metric_name
            if metric_tensor.ndim == 0:  # Scalar
                scalar_dict[new_name] = metric_tensor
            else:
                for i, cls in enumerate(self.classes):
                    scalar_dict[new_name+'_'+cls] = metric_tensor[i]

        self.log_dict(scalar_dict)


class LegacyModel(nn.Module):
    def __init__(self, learning_rate, class_weights=None, metrics=None):
        super(LegacyModel, self).__init__()
        self.learning_rate = learning_rate
        self.class_weights = class_weights
        self.metrics = metrics

    def fit(self, train_loader, epoch):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.train()
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = self(inputs)
            targets = torch.squeeze(targets)
            loss = nn.CrossEntropyLoss(weight=self.class_weights)(outputs, targets)
            total_loss += loss
            _, predictions = torch.max(torch.nn.Softmax(dim=1)(outputs.detach()), 1)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Epoch: {} {}/{} Training loss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(inputs),
                    len(train_loader.dataset),
                    loss
                ))

        print('Training loss: {:.6f}'.format(total_loss / len(train_loader)))

        return total_loss / len(train_loader)

    def evaluate(self, test_loader):
        self.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = self(inputs)
                targets = torch.squeeze(targets)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                total_loss += loss
                _, predictions = torch.max(torch.nn.Softmax(dim=1)(outputs), 1)
                correct += predictions.eq(targets.view_as(predictions)).sum()

        test_loss = total_loss/len(test_loader)

        test_percent_acc = 100. * correct / len(test_loader.dataset)
        print('Test loss: {:.6f}, Test accuracy: {}/{} ({:.1f}%)\n'.format(
            test_loss,
            correct,
            len(test_loader.dataset),
            test_percent_acc
        ))

        return test_percent_acc, test_loss

    def on_epoch_end(self, labels, preds):
        """
        Callbacks to perform at the end of an epoch
        """
        res = {}
        if self.metrics is not None:
            res['metrics'] = self.compute_metrics(labels, preds)

        return res

    def compute_metrics(self, labels, preds):
        """
        Callback to compute performance metrics
        :param labels: Torch tensor of labels
        :param preds: Torch tensor of model predictions
        """
        res = {}
        for metric in self.metrics:
            res[metric.name] = metric.compute(labels, preds)

        print('\n PERFORMANCE METRICS: \n')
        print(res)
        print('\n')

        return res


class CNN_MODEL(nn.Module):
    def __init__(self, model_cfg):
        super(CNN_MODEL, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=32, kernel_size=5)
        self.hidden1 = nn.Linear(20, 512)
        self.hidden2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(model_cfg['dropout'])
        self.output = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.hidden1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = self.dropout(x)
        x = F.relu(x)

        return self.output(x)


class MLP_MODEL_ITER2(nn.Module):
    def __init__(self, model_cfg):
        super(MLP_MODEL_ITER2, self).__init__()
        input_size = 40
        self.hidden1 = nn.Linear(input_size, 512)
        self.hidden2 = nn.Linear(256, 128)
        self.hidden3 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(model_cfg['dropout'])
        self.output = nn.Linear(64, 3)
        self.output_activation = torch.nn.Sigmoid()

        # def init_weights(m):
        #     if isinstance(m, nn.Linear):
        #         torch.nn.init.xavier_uniform_(m.weight)
        #         m.bias.data.fill_(0)
        #
        # self.apply(init_weights)

    def forward(self, x):
        x = torch.flatten(x, 1)

        x = self.hidden1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.hidden2(x)
        x = F.relu(x)

        x = self.hidden3(x)
        x = self.dropout(x)
        x = F.relu(x)

        x = self.output(x)
        x = self.output_activation(x)

        return x


class MLP_MODEL(nn.Module):
    def __init__(self, model_cfg):
        super(MLP_MODEL, self).__init__()
        input_size = 5
        self.hidden1 = nn.Linear(input_size, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.hidden3 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(model_cfg['dropout'])
        self.output = nn.Linear(32, 4)

        # def init_weights(m):
        #     if isinstance(m, nn.Linear):
        #         torch.nn.init.xavier_uniform_(m.weight)
        #         m.bias.data.fill_(0)
        #
        # self.apply(init_weights)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.hidden1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = self.dropout(x)
        x = self.hidden3(x)
        x = F.relu(x)

        return self.output(x)
