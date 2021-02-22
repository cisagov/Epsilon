# NOTICE
#
# This (software/technical data) was produced for the
# U. S. Government, under Contract Number HSHQDC-14-D-00006, and is
# subject to Federal Acquisition Regulation Clause 52.227-14, Rights
# in Data-General.  As prescribed in 27.409(b)(1), insert the
# following clause with any appropriate alternates:
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

import abc

from epsilon import monitor, buffers


class FilteredMonitor(monitor.Monitor):
    """
    A specialized monitor that filters detections before flagging input as invalid.
    """

    def __init__(self, receiver_id, monitor_timeout=None, threshold=0, min_detections=1, sample_window=1):
        """
        Create a FilteredMonitor; an alarm will not be set until at least @min_detections have occurred in @window
        samples

        @param receiver_id The receiver id
        @param monitor_timeout The maximum time allowed between samples
        @param threshold The monitor's threshold above which to trigger spoofing flags
        @param min_detections: The number of detections that must be reached before an alarm is returned
        @param sample_window: The number of most recent samples to check for min_detections
        """
        super(FilteredMonitor, self).__init__(receiver_id=receiver_id, monitor_timeout=monitor_timeout,
                                              threshold=threshold)

        if sample_window < min_detections:
            raise ValueError('The sample window must be at least as long as the number of detections to trigger.'
                             'Instead the window was %d while the number of detections was %s',
                             sample_window, min_detections)

        self._min_detections = min_detections
        self._status['alarm'] = False
        self._status['spoofing_flag'] = False

        self.detections = buffers.FIFORunningSum(max_size=sample_window)

    @abc.abstractmethod
    def _calculate_metric(self, message):
        """
        Subclasses will do monitor-specific processing here to determine if the given message is anomalous.

        @param message: Message to consume.
        @return: True if the current sample triggers an alarm and False otherwise.
        """

        pass

    def update(self, message):
        """
        Run the subclass monitor code (in _process_message) like normal but store the result in the running sum deque
        and only return True once the number of anomalous samples in it reaches self._min_detections.

        @param message: The message to consume.
        @return: True if the number of alarms from _process_message has passed self._min_detections.
        """

        # The update method in the Monitor base class returns True or False in lock step with _process_message)
        sample = super(FilteredMonitor, self).update(message)

        #  Break out early if the message wasn't usable
        if sample is None:
            return None

        # This is the raw, unfiltered spoofing determination for the last sample
        self._status['spoofing_flag'] = sample

        self.detections.append(int(sample))

        # Filter this subclass' update output using the m of n detector
        #  alarm is the filtered monitor spoofing determination
        if self.detections.sum >= self._min_detections:
            self._status['alarm'] = True
            return True

        self._status['alarm'] = False
        return False

    def reset(self):
        """
        Reset the filtered monitor by resetting the count of detections to zero.
        """

        super(FilteredMonitor, self).reset()

        self.detections.reset()
        self._status['spoofing_flag'] = False
