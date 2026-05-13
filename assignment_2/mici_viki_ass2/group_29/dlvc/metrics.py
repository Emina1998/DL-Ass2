from abc import ABCMeta, abstractmethod
import torch

class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        '''
        Resets internal state.
        '''

        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''

        pass

    @abstractmethod
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''

        pass


class SegMetrics(PerformanceMeasure):
    '''
    Mean Intersection over Union.
    '''

    def __init__(self, classes):
        self.classes = classes 
        
        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        self.conf_matrix = torch.zeros((len(self.classes),len(self.classes)), dtype = torch.int64)
        pass



    def update(self, prediction: torch.Tensor, 
               target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (b,c,h,w) where b=batchsize, c=num_classes, h=height, w=width.
        target must have shape (b,h,w) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        Make sure to not include pixels of value 255 in the calculation since those are to be ignored. 
        '''

       
        if prediction.ndim != 4 or target.ndim != 3:
            raise ValueError("Invalid shapes: prediction should be (b,c,h,w) shaped, target (b,h,w) ")
        
        
        
        b_dim, c_dim, h_dim, w_dim = prediction.shape
        b_dim_t, h_dim_t, w_dim_t = target.shape
        
        #print(f"C dim {c_dim}")
        #print(f"Class num: {len(self.classes)}")

        if c_dim != len(self.classes):
            raise ValueError("Number of classes in prediction does not match initialized value.")

        if b_dim != b_dim_t or h_dim != h_dim_t or w_dim_t != w_dim:
            raise ValueError("Shapes are unsupported")
        
        
        


        
        prediction_classes = torch.argmax(prediction, dim = 1)
        
        prediction_classes = prediction_classes.view(-1)
        target = target.view(-1)
        mask = target != 255
        prediction_classes = prediction_classes[mask]
        target = target[mask]
        
        for t,p in zip(target,prediction_classes):
            self.conf_matrix[t,p] += 1
        pass
   

    def __str__(self):
        '''
        Return a string representation of the performance, mean IoU.
        e.g. "mIou: 0.54"
        '''
        ##TODO implement
        return "mIoU: {}".format(self.mIoU())
        pass
          

    
    def mIoU(self) -> float:
        '''
        Compute and return the mean IoU as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        If the denominator for IoU calculation for one of the classes is 0,
        use 0 as IoU for this class.
        '''
        ##TODO implement
        ious = []
        row_sum = torch.sum(self.conf_matrix, dim = 1)
        col_sum = torch.sum(self.conf_matrix, dim = 0)
        ious = list(map(lambda i: 0 if row_sum[i] + col_sum[i] - self.conf_matrix[i][i] == 0  else self.conf_matrix[i][i]/(row_sum[i] + col_sum[i] - self.conf_matrix[i][i]),range(len(self.classes)))) 
        if len(ious) == 0:
            return  0.0
        
        return  sum(ious)/len(ious)

        
        pass





