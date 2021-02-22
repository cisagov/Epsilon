# -*- coding: utf-8 -*-
# NOTICE
#
# This (software/technical data) was produced for the U. S. Government
# under Contract Number HSHQDC-14-D-00006, and is subject to
# Federal Acquisition Regulation Clause 52.227-14, Rights in Data-General.
# As prescribed in 27.409(b)(1),
# insert the following clause with any appropriate alternates:
#
# Rights in Data-General (Deviation May 2014).
#
# No other use other than that granted to the U. S. Government,
# or to those acting on behalf of the U. S. Government under that Clause is
# authorized without the express written permission of The MITRE Corporation.
#
# For further information, please contact The MITRE Corporation,
# Contracts Management Office, 7515 Colshire Drive, McLean, VA 22102-7539,
# (703) 983-6000.
#
# Copyright 2017 The MITRE Corporation. All Rights Reserved.

"""
This file contains rolling buffer classes.
"""

import collections


class FIFORunningSum(object):
    """
    @brief A class that tracks the running sum of the previous N elements in a set of streaming data.

    This class provides a mechanism for tracking the running arithmetic sum of a stream of values using a First-In,
    First-Out (FIFO) queue structure with a specified maximum length. As new values are pushed into this object, the sum
    is updated. When the queue is full, pushing a new value will automatically remove the oldest sample.
    """

    def __init__(self, max_size):
        """
        @brief Creates a new FIFORunningSum object with specified max_size.

        @param[in] max_size The maximum number of elements in the buffer.
        """

        self._sum = 0
        self._buffer = collections.deque(maxlen=max_size)

    def __len__(self):
        """
        Get the number of elements in the FIFO queue

        @return: The number of elements in the FIFO queue
        """

        return len(self._buffer)

    def reset(self):
        """
        @brief Resets the FIFORunningSum object.

        This clears the buffer entirely and resets the sum.
        """

        self._buffer.clear()
        self._sum = 0

    def append(self, val):
        """
        @brief Adds a new value to the FIFORunningSum.

        @param[in] val The value to add.
        """

        if len(self._buffer) == self._buffer.maxlen:
            self._sum -= self._buffer[0]

        self._buffer.append(val)
        self._sum += val

    @property
    def sum(self):
        return self._sum

    @property
    def maxlen(self):
        return self._buffer.maxlen


class FIRFilter(object):
    """
    @brief Class implementing a finite-impulse-response (FIR) filter. An FIR filter has a series of coefficients, which
           are multiplied with a series of delayed inputs and then summed to give the overall filter response. The
           response can be written as follows:

    @f[
    y[n] = \sum_{i=0}^{N} a_i \cdot x[n-i]
    @f]

    Where @f$a_i@f$ are the filter coefficients: the first coefficient is for the MOST-RECENT sample and the last
    coefficient is for the OLDEST sample

    FIR filters are guaranteed to be stable for stable inputs and tend to have a linear phase response, leading to a
    constant filter delay for all frequencies (as opposed to a frequency dependent delay, which can occur in other
    filters). However, they may require more coefficients to achieve a comparable filtering response to other filters.
    """

    def __init__(self, coefficients):
        """
        @brief Initializes the FIR Filter with the given coefficients.

        @param coefficients The coefficients for the filter. The coefficients are in increasing order: the first
               element applies to the most recent sample and the last element applies to the oldest sample. See @ref
               coefficients for more information.
        """

        self._filter_length = len(coefficients)
        self._coefficients = coefficients
        self._past_samples = collections.deque(maxlen=len(coefficients))

        self._response = None

    @property
    def response(self):
        """
        Return the current response value of the filter or None if the filter is not initialized.
        """

        return self._response

    @property
    def coefficients(self):
        """
        The filter coefficients: the first coefficient goes with the MOST-RECENT sample and the last coefficient with the
        OLDEST sample
        """

        return self._coefficients

    @property
    def filter_length(self):
        """
        Return the length or size of the filter.
        """

        return self._filter_length

    @property
    def is_initialized(self):
        """
        Returns true if the filter has filled.
        """

        return len(self._past_samples) == self._filter_length

    def appendleft(self, sample):
        """
        @brief Update the filter with a new sample

        Called appendleft to mimic traditional filter diagrams; left-most element is the most-recent and multiplied by
        the first element in self._coefficients and so on.

        @param sample The new sample for the filter.
        """

        self._past_samples.appendleft(sample)

        # Return None as long as the filter has not finished initializing
        if self.is_initialized:
            # This is sequential deque access so should be O(1) just with a larger-than-desirable coefficient
            self._response = sum(element * coefficient for (element, coefficient) in zip(self._past_samples,
                                                                                         self._coefficients))
        else:   # This can happen if the filter has been reset
            self._response = None

    def reset(self):
        """
        @brief Resets the filter.

        This will remove all past samples and replace them with zeros, but
        does not modify the filter coefficients.
        """

        self._past_samples.clear()
        self._response = None

    def __len__(self):
        return len(self._past_samples)


class TimedBuffer(object):
    """
    @brief A buffer that stores time information with data, optionally ensuring that the samples represent a particular
           time duration and capping the age of samples

    The time information is unitless but expected to be consistent. The samples must be monotonically increasing.
    """

    def __init__(self, target_elapsed_time=float('inf'), keep_one_sample_before=True):
        """
        @brief Creates a new TimedBuffer.

        @param target_elapsed_time The approximate amount of time the buffer should represent; the buffer will keep
                one sample older than the window so that the span of time is at least this much (if there are enough
                samples).
        @param keep_one_sample_before If True, keep one sample before the oldest allowed based on target_elapsed_time
                so that the time window contained in the buffer is at least target_elapsed_time. If False, always
                evict samples that are older than target_elapsed_time; Default is to keep one sample before.
        """

        self._time_buffer = collections.deque()
        self._sample_buffer = collections.deque()

        self._target_elapsed_time = float('inf')
        self.target_elapsed_time = target_elapsed_time
        self._keep_one = keep_one_sample_before

    @property
    def target_elapsed_time(self):
        """
        @brief The maximum amount of time between the oldest and newest samples in the buffer; the oldest sample may
        be older than this window to ensure that self.elapsed_time >= target_elapsed_time
        """

        return self._target_elapsed_time

    @target_elapsed_time.setter
    def target_elapsed_time(self, val):
        if val < 0:
            raise ValueError('Expected max_elapsed_time to be non-negative')

        self._target_elapsed_time = val
        self.remove_old_samples()

    @property
    def newest_time(self):
        """
        @brief Returns time of the most recent sample in the buffer
        """

        if len(self) == 0:
            return 0

        return self._time_buffer[-1]

    @property
    def oldest_time(self):
        """
        @brief Returns time of the most recent sample in the buffer
        """

        if len(self) == 0:
            return 0

        return self._time_buffer[0]

    @property
    def elapsed_time(self):
        """
        @brief Returns the elapsed time in the buffer.

        The elapsed time is defined as the time between the newest and oldest sample. If the buffer is empty, this will
        be zero.

        @return The elapsed time in the buffer.
        """

        if len(self) < 2:
            return 0

        return self._time_buffer[-1] - self._time_buffer[0]

    @property
    def oldest_sample(self):
        """
        @brief Returns the oldest sample in the buffer.

        If there are no samples in the buffer, this returns None.

        @return The oldest sample in the buffer.
        """

        if len(self) == 0:
            return None

        return self._sample_buffer[0]

    @property
    def newest_sample(self):
        """
        @brief Returns the newest sample in the buffer.

        If there are no samples in the buffer, this returns None.

        @return The newest sample in the buffer.
        """

        if len(self) == 0:
            return None

        return self._sample_buffer[-1]

    def append(self, time, sample):
        """
        @brief Appends a new sample to the buffer.

        @param[in] time The time this sample occurs on.
        @param[in] sample The sample to add.
        """

        if time < self.newest_time:
            raise ValueError(
                'Time out or order: last time was {}, but the new time is {}.'.format(self.newest_time, time)
            )

        self._time_buffer.append(time)
        self._sample_buffer.append(sample)

        self.remove_old_samples()

    def remove_old_samples(self, keep_one_before=True):
        """
        @brief Removes old samples from the buffer.

        @param[in] keep_one_before Optionally override the keep_one_sample_before parameter in __init__: If True (the
                   default) keep one sample older than the target elapsed window so that the full time window is
                   present; if False remove all old samples regardless of the resulting window size.
        """

        if len(self._time_buffer) < 1:
            return

        # Figure out how many stale elements there are without altering the data structures
        num_to_pop = 0
        target_time = self._time_buffer[-1] - self._target_elapsed_time
        for sample_time in self._time_buffer:
            if sample_time >= target_time:
                break

            num_to_pop += 1

        # Keep one sample before, if desired
        if keep_one_before:
            num_to_pop -= 1

        # Pop without referencing the indices or size of the structures (if num_to_pop is 0 or -1 this will do nothing)
        for _ in range(0, num_to_pop, 1):
            self.popleft()

    def popleft(self):
        """
        Remove the oldest element from the buffer
        """

        self._time_buffer.popleft()
        self._sample_buffer.popleft()

    def __len__(self):
        """
        @brief Returns the length of the buffer.
        """

        return len(self._time_buffer)

    def reset(self):
        """
        Clear the underlying buffers
        """

        self._time_buffer.clear()
        self._sample_buffer.clear()

    def times(self):
        return self._time_buffer


class TimedRunningSum(TimedBuffer):
    """
    A timed buffer that keeps a running sum of its contents
    """

    def __init__(self, target_elapsed_time=float('inf')):
        super(TimedRunningSum, self).__init__(target_elapsed_time=target_elapsed_time)

        self._sum = 0

    @property
    def sum(self):
        return self._sum

    def reset(self):
        super(TimedRunningSum, self).reset()
        self._sum = 0

    def append(self, time, sample):
        super(TimedRunningSum, self).append(time, sample)
        self._sum += sample

    def popleft(self):
        self._sum -= self.oldest_sample

        super(TimedRunningSum, self).popleft()
