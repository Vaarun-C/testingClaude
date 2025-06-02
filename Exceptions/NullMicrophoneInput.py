class NullMicrophoneInputException(Exception):
    """
    Exception raised for no microphone input.
    """
    def __init__(self):
        super().__init__('No Microphone input')