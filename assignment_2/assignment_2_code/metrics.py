from abc import ABCMeta, abstractmethod
import torch

class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class SegMetrics(PerformanceMeasure):
    '''
    Mean Intersection over Union.
    '''

    def __init__(self, classes):
        self.classes = classes
        self.reset()

    def reset(self) -> None:
        self.conf_matrix = torch.zeros((len(self.classes), len(self.classes)), dtype=torch.int64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        if prediction.ndim != 4 or target.ndim != 3:
            raise ValueError("prediction must be (b,c,h,w) and target must be (b,h,w)")
        
        b, c, h, w = prediction.shape
        bt, ht, wt = target.shape

        if c != len(self.classes):
            raise ValueError("Number of classes in prediction does not match initialized value.")
        
        if b != bt or h != ht or w != wt:
            raise ValueError("Batch size or spatial dimensions do not match between prediction and target.")

        pred_classes = torch.argmax(prediction, dim=1).view(-1)
        target = target.view(-1)

        mask = target != 255
        pred_classes = pred_classes[mask]
        target = target[mask]

        for t, p in zip(target, pred_classes):
            self.conf_matrix[t, p] += 1

    def __str__(self) -> str:
        return "mIoU: {}".format(self.mIoU())

    def mIoU(self) -> float:
        row_sum = torch.sum(self.conf_matrix, dim=1)
        col_sum = torch.sum(self.conf_matrix, dim=0)

        ious = []
        for i in range(len(self.classes)):
            denom = row_sum[i] + col_sum[i] - self.conf_matrix[i][i]
            if denom == 0:
                ious.append(0.0)
            else:
                ious.append((self.conf_matrix[i][i] / denom).item())

        if len(ious) == 0:
            return 0.0
        
        return sum(ious) / len(ious)