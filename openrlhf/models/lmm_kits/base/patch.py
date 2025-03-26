from abc import ABC, abstractmethod

class BasePatch(ABC):
    def __init__(self):
        self.loaded = False
    @abstractmethod
    def _add_get_inputs_embeds():
        '''
        Add a `get_inputs_embeds(*args,**kwargs)` method to the model class, 
        which embeds image embeddings into the text embeddings and return the results.
        '''
        return NotImplementedError

    @abstractmethod
    def _add_get_position_ids():
        '''
        Add a `get_posiiton_ids(*args,**kwargs)` method to the model class, 
        which return the position_ids of the given inputs.
        '''
        return NotImplementedError

    @abstractmethod
    def _add_offset_split_position_ids():
        '''
        Add a `offset_split_position_ids(*args,**kwargs)` method to the model class, 
        which offset the split position_ids to true position_ids.
        '''
        return NotImplementedError

    @classmethod
    @abstractmethod
    def _load_all_patches(cls):
        '''
        Load all patches.
        '''
        return NotImplementedError

    def load_all_patches(self):
        if not self.loaded:
            self._load_all_patches()
            self.loaded = True
