''' Metric class for tracking correlations by saving predictions '''
import logging
import os
import numpy as np
import pickle
import torch
from datetime import datetime

from allennlp.training.metrics.metric import Metric
from overrides import overrides
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef

from seqeval.metrics import accuracy_score
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import recall_score

logger = logging.getLogger(__name__)


@Metric.register("fastMatthews")
class FastMatthews(Metric):
    """Fast version of Matthews correlation.
    Computes confusion matrix on each batch, and computes MCC from this when
    get_metric() is called. Should match the numbers from the Correlation()
    class, but will be much faster and use less memory on large datasets.
    """

    def __init__(self, n_classes=2):
        assert n_classes >= 2
        self.n_classes = n_classes
        self.reset()
        self.corr_type = 'matthews'

    def __call__(self, predictions, labels):
        # Convert from Tensor if necessary
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        assert predictions.dtype in [np.int32, np.int64, int]
        assert labels.dtype in [np.int32, np.int64, int]

        C = confusion_matrix(labels.ravel(), predictions.ravel(),
                             labels=np.arange(self.n_classes, dtype=np.int32))
        assert C.shape == (self.n_classes, self.n_classes)
        self._C += C

    def mcc_from_confmat(self, C):
        # Code below from
        # https://github.com/scikit-learn/scikit-learn/blob/ed5e127b/sklearn/metrics/classification.py#L460
        t_sum = C.sum(axis=1, dtype=np.float64)
        p_sum = C.sum(axis=0, dtype=np.float64)
        n_correct = np.trace(C, dtype=np.float64)
        n_samples = p_sum.sum()
        cov_ytyp = n_correct * n_samples - np.dot(t_sum, p_sum)
        cov_ypyp = n_samples ** 2 - np.dot(p_sum, p_sum)
        cov_ytyt = n_samples ** 2 - np.dot(t_sum, t_sum)
        mcc = cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)

        if np.isnan(mcc):
            return 0.
        else:
            return mcc

    def get_metric(self, reset=False):
        # Compute Matthews correlation from confusion matrix.
        # see https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
        correlation = self.mcc_from_confmat(self._C)
        if reset:
            self.reset()
        return correlation

    @overrides
    def reset(self):
        self._C = np.zeros((self.n_classes, self.n_classes),
                           dtype=np.int64)


@Metric.register("correlation")
class Correlation(Metric):
    """Aggregate predictions, then calculate specified correlation"""

    def __init__(self, corr_type):
        self._predictions = []
        self._labels = []
        if corr_type == 'pearson':
            corr_fn = pearsonr
        elif corr_type == 'spearman':
            corr_fn = spearmanr
        elif corr_type == 'matthews':
            corr_fn = matthews_corrcoef
        else:
            raise ValueError("Correlation type not supported")
        self._corr_fn = corr_fn
        self.corr_type = corr_type

    def _correlation(self, labels, predictions):
        corr = self._corr_fn(labels, predictions)
        if self.corr_type in ['pearson', 'spearman']:
            corr = corr[0]
        return corr

    def __call__(self, predictions, labels):
        """ Accumulate statistics for a set of predictions and labels.
        Values depend on correlation type; Could be binary or multivalued.
        This is handled by sklearn.
        Args:
            predictions: Tensor or np.array
            labels: Tensor or np.array of same shape as predictions
        """
        # Convert from Tensor if necessary
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        # Verify shape match
        assert predictions.shape == labels.shape, ("Predictions and labels "
                                                   "must have matching shape. "
                                                   "Got:"
                                                   " preds=%s, labels=%s" % (
                                                       str(predictions.shape),
                                                       str(labels.shape)))
        if self.corr_type == 'matthews':
            assert predictions.dtype in [np.int32, np.int64, int]
            assert labels.dtype in [np.int32, np.int64, int]

        predictions = list(predictions.flatten())
        labels = list(labels.flatten())

        self._predictions += predictions
        self._labels += labels

    def get_metric(self, reset=False):
        correlation = self._correlation(self._labels, self._predictions)
        if reset:
            self.reset()
        return correlation

    @overrides
    def reset(self):
        self._predictions = []
        self._labels = []


@Metric.register('mrr')
class MeanReciprocalRank(Metric):
    def __init__(self):
        self._sum = 0.0
        self._n = 0.0

    def __call__(self, predictions, labels, mask):
        # Flatten
        labels = labels.view(-1)
        mask = mask.view(-1).float()
        predictions = predictions.view(labels.shape[0], -1)

        # MRR computation
        label_scores = predictions.gather(-1, labels.unsqueeze(-1))
        rank = predictions.ge(label_scores).sum(1).float()
        reciprocal_rank = 1 / rank
        self._sum += (reciprocal_rank * mask).sum().item()
        self._n += mask.sum().item()

    def get_metric(self, reset=False):
        mrr = self._sum / (self._n + 1e-13)
        if reset:
            self.reset()
        return mrr

    @overrides
    def reset(self):
        self._sum = 0.0
        self._n = 0.0


@Metric.register('microf1')
class MicroF1(Metric):
    def __init__(self, negative_label: int):
        """
        Micro-averaged F1 score

        Parameters
        ==========
        negative_label : ``int``
            Index of negative class label.
        """
        self._negative_label = negative_label
        self._tp = 0
        self._fp = 0
        self._fn = 0

    def __call__(self, predictions, labels, mask=None):
        if mask is None:
            mask = torch.ones_like(labels, dtype=torch.uint8)

        mask = mask.detach().cpu().numpy()

        gold_negative = labels.eq(self._negative_label).detach().cpu().numpy()
        pred_negative = predictions.eq(
            self._negative_label).detach().cpu().numpy()

        correct = (predictions == labels).detach().cpu().numpy()
        incorrect = (predictions != labels).detach().cpu().numpy()

        tp = correct & mask & ~gold_negative & ~pred_negative
        fp = incorrect & mask & ~pred_negative
        fn = incorrect & mask & ~gold_negative

        self._tp += tp.astype(np.int).sum()
        self._fp += fp.astype(np.int).sum()
        self._fn += fn.astype(np.int).sum()

    def get_metric(self, reset=False):
        precision = self._tp / (self._tp + self._fp + 1e-13)
        recall = self._tp / (self._tp + self._fn + 1e-13)
        f1 = 2 * precision * recall / (precision + recall + 1e-13)
        if reset:
            self.reset()
        return precision, recall, f1

    @overrides
    def reset(self):
        self._tp = 0
        self._fp = 0
        self._fn = 0


@Metric.register('seqeval')
class SeqEval(Metric):
    def __init__(self, label_map, all_pred_f=None, all_lab_f=None):
        """ args:
                label_map: a dict of the form {"label": index}, the same one
                    that was used to numericalize the labels in the first
                    place.
                all_pred_f=None: name of file in which to store all
                    predictions. Each batch will reload the file and append
                    the current batch. Particularly useful for doing stats
                    on an entire corpus. If None, will create a timestamped
                    file.
                all_lab_f=None: name of file in which to store all labels.
                    Each batch will reload the file and append the current
                    batch. Particularly useful for doing stats on an entire
                    corpus. If None, will create a timestamped file.
        """
        self.rec = 0.
        self.f1 = 0.
        self.prec = 0.
        self.label_map = {v: k for k, v in label_map.items()}
        self.all_pred_f = all_pred_f
        self.all_lab_f = all_lab_f

        date = list(datetime.now().timetuple())[:6]
        timestamp = '_'.join([str(i) for i in date]) + '_'
        if all_pred_f is None:
            self.all_pred_f = timestamp + 'seqeval_pred.out'
        if all_lab_f is None:
            self.all_lab_f = timestamp + 'seqeval_lab.out'

    def __call__(self, predictions, labels, mask=None):
        """ Args:
                predictions: list of predicted label ids (one per subtoken)
                labels: list of true label ids (one per subtoken)
                mask: The attention mask. List of booleans or equivalent
                    0s/1s, one per subtoken, with True for non-padding
                    tokens and False for padding tokens.
        """
        if mask is None:
            mask = torch.ones_like(labels, dtype=torch.uint8)
        mask = mask.tolist()

        assert predictions.size() == labels.size()

        predictions = self.to_IOB2(predictions.int().tolist(), mask)
        labels = self.to_IOB2(labels.int().tolist(), mask)

        # currently our predictions and labels are wordpiece-based
        predictions, labels = self.wordpiece_to_word(predictions, labels)

        # if requested, handle the cross-batch prediction/label backups
        if self.all_pred_f is not None:
            try:
                with open(self.all_pred_f, 'rb') as f:
                    allpred = pickle.load(f)
                allpred.extend(predictions)
            except FileNotFoundError:
                allpred = predictions
            outDir:str = os.path.dirname(self.all_pred_f)
            if outDir is not "":
                os.makedirs(outDir, exist_ok=True)
            with open(self.all_pred_f, 'wb') as f:
                pickle.dump(allpred, f)
        if self.all_lab_f is not None:
            try:
                with open(self.all_lab_f, 'rb') as f:
                    alllab = pickle.load(f)
                alllab.extend(labels)
            except FileNotFoundError:
                alllab = labels
            outDir:str = os.path.dirname(self.all_lab_f)
            if outDir is not "":
                os.makedirs(outDir, exist_ok=True)
            with open(self.all_lab_f, 'wb') as f:
                pickle.dump(alllab, f)

        try:
            self.prec = precision_score(labels, predictions)
            self.rec = recall_score(labels, predictions)
            self.f1 = f1_score(labels, predictions)
        except ValueError as e:
            logger.error("Error: could not compute Precision/"
                         "Recall/F1. Too few classes.")
            logger.error(f"labels: {labels}")
            logger.error(f"predictions: {predictions}")
            raise e

        self.acc = accuracy_score(labels, predictions)

    def get_metric(self, reset=False):
        ret_values = self.acc, self.prec, self.rec, self.f1
        if reset:
            self.reset()
        return ret_values

    @overrides
    def reset(self):
        self.rec = 0.
        self.f1 = 0.
        self.prec = 0.

    def to_IOB2(self, taglist, mask):
        """ Takes predictions of the form [[1, 2, 1], [1, 3, ...]]
            and returns an IOB2-formatted prediction with no padding.
            Args:
                pred (Tensor): the model's predictions for the sequence
        """

        # Currently, taglist contains label IDs instead of actual labels,
        # since that's what the model understands. Here, we're switching
        # the IDs with the labels themselves.
        taglist = [[p for j, p in enumerate(seq) if mask[i][j]]
                   for i, seq in enumerate(taglist)]
        taglist = [[self.label_map[tok] for tok in seq] for seq in taglist]
        return taglist

    def wordpiece_to_word(self, predictions, labels):
        """ Remove predictions that correspond to non-token-starting
            wordpieces. The corresponding labels are [PAD] labels.
        """
        # remove predictions that correspond to padding tokens or to
        # non-token-starting wordpieces
        predictions = [[p for j, p in enumerate(seq)
                        if labels[i][j] != '[PAD]']
                       for i, seq in enumerate(predictions)]
        # The model may have predicted incorrectly that some real tokens
        # are padding tokens. This would break the evaluation, so we naively
        # assume that these would otherwise be labeled 'O'. This is definitely
        # wrong and suboptimal for performance, but it's simple.
        predictions = [[p if p != '[PAD]' else 'O' for p in seq]
                       for seq in predictions]
        # remove labels that correspond to padding tokens or to
        # non-token-starting wordpieces
        labels = [[lab for lab in seq
                   if lab != '[PAD]']
                  for seq in labels]
        return predictions, labels
