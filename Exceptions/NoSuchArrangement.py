class NoSuchArrangementException(Exception):
    """
    Exception raised for no available microphone arrangemnt.
    """
    def __init__(self):
        super().__init__('No equation available for arrangement')