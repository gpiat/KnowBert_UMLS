import numpy as np
import torch

from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.nn.util import batched_index_select
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics import F1Measure
from allennlp.training.metrics import Metric
from kb.common import F1Metric
from kb.common import get_dtype_for_module
from kb.evaluation.fbeta_measure import FBetaMeasure
from kb.evaluation.semeval2010_task8 import SemEval2010Task8Metric
from kb.metrics import Correlation
from kb.metrics import FastMatthews
from kb.metrics import MicroF1
from kb.metrics import SeqEval


@Model.register('simple-classifier')
class SimpleClassifier(Model):
    def __init__(self, vocab: Vocabulary,
                 model: Model,
                 task: str,
                 num_labels: int,
                 bert_dim: int,
                 metric_a: Metric,
                 metric_b: Metric = None,
                 concat_word_a_b: bool = False,
                 concat_word_a: bool = False,
                 include_cls: bool = True,
                 dropout_prob: float = 0.1,
                 use_bce_loss: bool = False):

        super().__init__(vocab)
        assert task == 'regression' or task == 'classification'\
            or 'word_classification', task
        if task == 'regression':
            assert num_labels == 1
        self.task = task
        self.num_labels = num_labels
        self.metrics = []

        if not include_cls:
            assert concat_word_a_b or concat_word_a or\
                task == 'word_classification'
        assert not (concat_word_a_b and concat_word_a)

        if concat_word_a_b and include_cls:
            classifier_dim = bert_dim * 3
        elif concat_word_a_b and not include_cls:
            classifier_dim = bert_dim * 2
        elif concat_word_a and include_cls:
            classifier_dim = bert_dim * 2
        elif concat_word_a and not include_cls:
            classifier_dim = bert_dim
        else:
            classifier_dim = bert_dim

        self.concat_word_a_b = concat_word_a_b
        self.concat_word_a = concat_word_a
        self.include_cls = include_cls

        self.dropout = torch.nn.Dropout(dropout_prob)
        self.classifier = torch.nn.Linear(classifier_dim, num_labels)

        if metric_a is not None:
            self.metrics.append(metric_a)
        if metric_b is not None:
            self.metrics.append(metric_b)

        self.model = model

        if task == 'classification' or task == 'word_classification':
            if use_bce_loss:
                self.loss = torch.nn.BCEWithLogitsLoss()
                assert len(self.metrics) == 1
            else:
                self.loss = torch.nn.CrossEntropyLoss()
        else:
            self.loss = torch.nn.MSELoss()
        self.use_bce_loss = use_bce_loss

    def forward(self,
                tokens=None,
                segment_ids=None,
                candidates=None,
                label_ids=None,
                **kwargs):
        model_output = self.model(tokens=tokens, segment_ids=segment_ids,
                                  candidates=candidates,
                                  lm_label_ids=None,
                                  next_sentence_label=None)
        # model_output has keys
        # ['umls', 'loss', 'pooled_output', 'contextual_embeddings']
        # model_output['pooled_output'] is the CLS token for each batch
        #     and has shape [batch_size, bert_dim]
        # model_output['contextual_embeddings'] is the actual word
        #     predictions and has shape [batch_size, ntokens, bert_dim])
        if self.concat_word_a_b:
            # concat the selected words in index_a, index_b
            # (batch_size, timesteps, dim)
            contextual_embeddings = model_output['contextual_embeddings']
            word_a = batched_index_select(
                contextual_embeddings, kwargs['index_a'])
            word_b = batched_index_select(
                contextual_embeddings, kwargs['index_b'])
            if self.include_cls:
                pooled_output = torch.cat([model_output['pooled_output'],
                                           word_a,
                                           word_b], dim=-1)
            else:
                pooled_output = torch.cat([word_a, word_b], dim=-1)
        elif self.concat_word_a:
            contextual_embeddings = model_output['contextual_embeddings']
            word_a = batched_index_select(
                contextual_embeddings, kwargs['index_a'])
            if self.include_cls:
                pooled_output = torch.cat([model_output['pooled_output'],
                                           word_a], dim=-1)
            else:
                pooled_output = word_a
        else:
            pooled_output = model_output['pooled_output']

        if self.task == 'word_classification':
            # here logits has shape [batch_size, ntokens, num_classes]
            logits = self.classifier(self.dropout(
                model_output['contextual_embeddings']))
        else:
            # classifier(pooled_output) has size
            #    [batch_size, seq_size, num_labels]
            # after .view(...) it has size [batch_size x seq_size, num_labels]
            logits = self.classifier(self.dropout(
                pooled_output)).view(-1, self.num_labels)

        outputs = {}
        if self.task == 'classification':
            # or self.task == 'word_classification':
            outputs['logits'] = logits.detach()
            if self.use_bce_loss:
                loss = self.loss(logits, label_ids.to(
                    get_dtype_for_module(self)))
                outputs['predictions'] = None
            else:
                _, predictions = torch.max(logits, -1)
                outputs['predictions'] = predictions.detach()
                loss = self.loss(logits, label_ids.view(-1))
            outputs['loss'] = loss
        elif self.task == 'word_classification':
            outputs['logits'] = logits.detach()
            # this is actually an argmax, predictions has shape
            #    [batch_size, seq]
            _, predictions = torch.max(outputs['logits'], -1)
            outputs['predictions'] = predictions.detach()
            # label_ids also has shape [batch_size, seq]
            try:
                loss = self.loss(logits.permute(0, 2, 1),
                                 label_ids)
            except ValueError as e:
                print(f"loss: {self.loss}")
                print(f"predictions shape: {predictions.size()}")
                print(f"logits shape: {logits.size()}")
                print(f"label_ids shape: {label_ids.size()}")
                raise e
            outputs['loss'] = loss
        else:
            labels = label_ids.to(self.classifier.weight.dtype)
            outputs['loss'] = self.loss(logits, labels)
            outputs['predictions'] = logits.detach()

        if self.use_bce_loss:
            batch_size, num_classes = outputs['logits'].shape
            batches = np.arange(batch_size).repeat(
                num_classes).reshape(batch_size, num_classes)
            classes = np.arange(num_classes).repeat(
                batch_size).reshape(num_classes, batch_size).T
            # batches = [[0, 0, 0], [1, 1, 1], ...]
            # classes = [[0, 1, 2, 3, ..], [0, 1, 2, 3, ..]]

            # threshold logits at 0 to only get predictions with values > 0
            np_logits = outputs['logits'].cpu().numpy()
            predicted_logits = np_logits > 0
            predicted = list(
                zip(batches[predicted_logits], classes[predicted_logits]))

            # actual predictions
            actual_classes = label_ids.cpu().numpy() > 0
            actual = list(
                zip(batches[actual_classes], classes[actual_classes]))

            # actual and predicted are lists of tuples
            # (batch_index, class_index)
            self.metrics[0]([predicted], [actual])

        else:
            for metric in self.metrics:
                if isinstance(metric, (CategoricalAccuracy,
                                       F1Measure,
                                       FBetaMeasure)):
                    metric(outputs['logits'], label_ids)
                elif isinstance(metric, SemEval2010Task8Metric):
                    metric(outputs['predictions'].view(-1), label_ids.view(-1))
                elif isinstance(metric, (FastMatthews, Correlation,
                                         MicroF1)):
                    metric(outputs['predictions'], label_ids)
                elif isinstance(metric, SeqEval):
                    try:
                        metric(outputs['predictions'],
                               label_ids,
                               mask=tokens['tokens'] != 0)
                    except RuntimeError as e:
                        print("predictions shape: "
                              f"{outputs['predictions'].size()}")
                        print("label_ids shape: "
                              f"{label_ids.size()}")
                        print(f"tokens type: {type(tokens['tokens'])}")
                        mask = tokens['tokens'] != 0
                        print("mask shape: "
                              f"{mask.size()}")
                        print("mask: "
                              f"{mask}")
                        raise e
                else:
                    raise Exception(f'Unsupported metric {metric}.')

        return {'loss': outputs['loss'], 'predictions': outputs['predictions'],
                'logits': logits.detach()}

    def get_metrics(self, reset: bool = False):
        metrics = {}
        avg_metric = None
        for metric in self.metrics:
            if isinstance(metric, CategoricalAccuracy):
                metrics['accuracy'] = metric.get_metric(reset)
                m = metrics['accuracy']
            elif isinstance(metric, F1Measure):
                metrics['f1'] = metric.get_metric(reset)[-1]
                m = metrics['f1']
            elif isinstance(metric, FBetaMeasure):
                metrics['f1'] = metric.get_metric(reset)['fscore']
                m = metrics['f1']
            elif isinstance(metric, FastMatthews) or\
                    isinstance(metric, Correlation):
                metrics[f'correlation_{metric.corr_type}'] =\
                    metric.get_metric(reset)
                m = metrics[f'correlation_{metric.corr_type}']
            elif isinstance(metric, SemEval2010Task8Metric):
                metrics['f1'] = metric.get_metric(reset)
                m = metrics['f1']
            elif isinstance(metric, MicroF1):
                precision, recall, f1 = metric.get_metric(reset)
                metrics['micro_prec'] = precision
                metrics['micro_rec'] = recall
                metrics['micro_f1'] = f1
                m = metrics['micro_f1']
            elif isinstance(metric, F1Metric):
                precision, recall, f1 = metric.get_metric(reset)
                metrics['precision'] = precision
                metrics['recall'] = recall
                metrics['f1'] = f1
                m = metrics['f1']
            elif isinstance(metric, SeqEval):
                accuracy, precision, recall, f1 =\
                    metric.get_metric(reset)
                # , report =\
                metrics['accuracy'] = accuracy
                metrics['precision'] = precision
                metrics['recall'] = recall
                metrics['f1'] = f1
                # print(report)
                m = metrics['f1']
            else:
                raise ValueError

            if avg_metric is None:
                avg_metric = [m, 1.0]
            else:
                avg_metric[0] += m
                avg_metric[1] += 1
        metrics = {k: float(v) for k, v in metrics.items()}
        if avg_metric is not None:
            metrics["avg_metric"] = avg_metric[0] / avg_metric[1]
        return metrics
