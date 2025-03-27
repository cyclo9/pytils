import numpy as np


class CircleBuffer:
    def __init__(self, maxlen, initial_values=None, dtype=float):
        """
        Initialize a circular buffer with a fixed maximum length and optional initial values.

        Args:
            maxlen: Maximum number of elements the buffer can hold.
            initial_values: Optional array-like object to initialize the buffer (default: None).
            dtype: Data type for the underlying NumPy array (default: float).
        """
        if not isinstance(maxlen, int) or maxlen <= 0:
            raise ValueError("maxlen must be a positive integer")

        self.maxlen = maxlen
        self.buffer = np.zeros(maxlen, dtype=dtype)
        self.start = 0
        self.size = 0

        # Handle initial values if provided
        if initial_values is not None:
            # Convert to NumPy array and flatten to 1D
            values = np.asarray(initial_values, dtype=dtype).ravel()
            if len(values) > 0:
                # If more values than maxlen, take the last maxlen elements
                if len(values) > self.maxlen:
                    values = values[-self.maxlen :]
                # Write values into the buffer
                self.buffer[: len(values)] = values
                self.size = len(values)

    def append(self, value):
        """
        Append a value to the buffer, overwriting the oldest element if full.

        Args:
            value: Scalar value to append.
        """
        # Calculate the index where the new value goes (end of the buffer)
        end = (self.start + self.size) % self.maxlen
        # Store the value
        self.buffer[end] = value

        # Update size and start pointer
        if self.size < self.maxlen:
            self.size += 1  # Buffer isn't full yet, just increase size
        else:
            self.start = (self.start + 1) % self.maxlen  # Full, move start forward

    def as_array(self):
        """
        Return the buffer contents as a NumPy array in chronological order.
        """
        if self.size == 0:
            return np.array([], dtype=self.buffer.dtype)

        if self.start + self.size <= self.maxlen:
            # Data is contiguous, no wrap-around
            return self.buffer[self.start : self.start + self.size].copy()
        else:
            # Data wraps around, concatenate the two parts
            return np.concatenate(
                (
                    self.buffer[self.start :],
                    self.buffer[: self.start + self.size - self.maxlen],
                )
            )

    def get(self, idx=-1):
        return self.as_array()[idx]

    def __len__(self):
        return self.size

    def __str__(self):
        return f"CircularArray({self.as_array()}, maxlen={self.maxlen})"
