from abc import ABC, abstractmethod

class VLM_Model(ABC):

    
    @abstractmethod
    def infer(prompt, frames_info, dynamic_resolution_config = None):
        r"""
            return : str
        """
        pass
