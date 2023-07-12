from augment import get_augment

augment = get_augment()

def sample_two_views(x):
    x1 = augment(x)
    x2 = augment(x)
    return x1, x2

class ViewTransform(object):
    
    # TODO implement this __call__ function to create 
    # two views from the image x and return them as tuple
    def __call__(self, x):
        return sample_two_views(x)
    
    
