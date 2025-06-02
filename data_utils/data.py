def convert_to_24bit(byte_data: bytes) -> tuple:
    """
    Convert a 3-byte array to a 24-bit integer and separate the first bit.
    
    Parameters:
    - byte_data (bytes): A sequence of 3 bytes to be converted to a 24-bit integer.
    
    Returns:
    - tuple: A tuple containing:
        - The 23-bit value (int).
        - The first bit (int), separated from the 24-bit integer.
    
    Raises:
    - ValueError: If the input is not exactly 3 bytes.
    """
    if len(byte_data) != 3:
        raise ValueError("Input must be exactly 3 bytes.")
    
    # Combine the 3 bytes into a 24-bit integer
    value_24bit = (byte_data[0] << 16) | (byte_data[1] << 8) | byte_data[2]
    
    # Extract the first bit (most significant bit)
    first_bit = (value_24bit >> 23) & 1
    
    # Extract the remaining 23 bits
    remaining_23_bits = value_24bit & 0x7FFFFF  # 0x7FFFFF is a mask to keep the last 23 bits

    return remaining_23_bits, first_bit


def split_into_chunks(data: bytes, chunk_size: int = 12) -> list:
    """
    Split the data into chunks of a specified size.
    
    Parameters:
    - data (bytes): The data to be split into chunks.
    - chunk_size (int, optional): The size of each chunk. Default is 12 bytes.
    
    Returns:
    - list: A list of chunks, where each chunk is a sequence of `chunk_size` bytes.
    """
    # Split data into chunks of the specified size
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def get_channel_data(data: list, channel: int = 1) -> list:
    """
    Extract signal values for a specified channel from a list of split data chunks.
    
    Parameters:
    - split_data (list): A list of data chunks to extract values from.
    - channel (int, optional): The channel index (1-based). Default is 1 (first channel).
    
    Returns:
    - list: A list of signal values, scaled and adjusted for the specified channel.
    """
    data = data[40:]
    data = split_into_chunks(data)
    
    lsb_scaling_factor = 0.000000298  # Least Significant Bit scaling factor (unit conversion)
    signal = []
    chunk_size = 3  # Each signal is made of 3 bytes

    # Determine the byte position for the specified channel
    channel_data_end = channel * chunk_size

    # Process each chunk in the split data
    for data in data:
        # Extract the channel's 3-byte data and convert it to a 24-bit value and first bit
        value, sign = convert_to_24bit(data[channel_data_end - chunk_size:channel_data_end])
        
        # Apply the LSB scaling factor to the value
        value *= lsb_scaling_factor
        
        # Adjust the value based on the first bit (sign)
        if sign == 0:
            value += 2.5  # If the sign bit is 0, adjust the value by adding 2.5
        
        signal.append(value)

    return signal
