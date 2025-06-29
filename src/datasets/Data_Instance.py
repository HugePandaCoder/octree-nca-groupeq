

class Data_Container():
    r"""The data container stores all the data after it has been loaded and preprocessed, so that this 
        only needs to be done once.
    """

    def __init__(self) -> None:
        self.data = {}
        #print("DATA INSTANCE CREATED")

    def get_data(self, key: str) -> tuple | bool:
        r"""Try to get data that has already been processed
            #Args
                key: The key of the data
                data: The data to be loaded
        """
        if key in self.data:
            return self.data[key]
        else: 
            return False

    def set_data(self, key: str, data: tuple) -> tuple:
        r"""Store data to load later
            #Args
                key: The key of the data
                data: The data to be stored
        """
        self.data[key] = data