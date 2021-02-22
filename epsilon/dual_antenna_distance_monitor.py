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

import math
import numpy as np

from epsilon import buffers, monitor


class StubPosMonitor(monitor.Monitor):
    """
    Helper class that just adds a TimedRunningSum buffer to a generic monitor that specifically looks for
    the ecef_position message element.
    """

    def __init__(self, receiver_id, monitor_timeout=None, time_window=0):
        super(StubPosMonitor, self).__init__(receiver_id=receiver_id, monitor_timeout=monitor_timeout,
                                             threshold=float('inf'))

        self.samples = buffers.TimedRunningSum(target_elapsed_time=time_window)

    def _calculate_metric(self, message):
        if 'ecef_position' not in message:
            return None

        time = message['rxTime']
        position = np.array(message['ecef_position'])

        # This monitor does NOT want one sample before the window
        self.samples.append(time, position)
        if self.samples.elapsed_time > self.samples.target_elapsed_time:
            self.samples.popleft()

        return position

    def reset(self):
        super(StubPosMonitor, self).reset()
        self.samples.reset()


class DualAntennaDistanceMonitor(monitor.Monitor):
    """
    @brief Monitors the position solution from two nearby receivers to identify spoofing.

    This monitor identifies when both receivers are fully captured by a single spoofer by identifying when their
    position solutions are drawn from distributions with the same mean.

    This monitor takes advantage of the fact that a spoofer will have difficulty targeting a small geographic area for
    its attack. If multiple receivers are present in the area of effect, then it is likely that all will be
    captured. With a single spoofer, this will cause each of the fully captured receivers to report that it is at a
    single spoofed location.

    @section Algorithm Description
    This algorithm computes the average location of two receivers and computes the distance between them. If this is too
    small, an alarm is triggered.  @section Receiver Separation Preliminary testing shows that the receivers likely
    should be around two standard deviations away from each other. This likely will correspond to some distance between
    10-20 meters.

    @section Monitor Limitations
    This monitor will only detect spoofing if both receivers are fully captured. If either receiver is not/partially
    captured, this monitor will not trigger. Other algorithms are needed to identify these partial capture conditions.
    """

    def __init__(self, receiver_id_1, receiver_id_2, monitor_timeout=None, minimum_samples=10, threshold=2,
                 time_range=15):
        """
        @brief Creates a new monitor.

        @param receiver_id_1 The ID for the first receiver to process events from.
        @param receiver_id_2 The ID of the second receiver to process events from.
        @param threshold The maximum distance between receivers to trigger the monitor, in meters.
        @param time_range The number of seconds over which to examine position solutions. This must be a positive
               number.
        """

        super(DualAntennaDistanceMonitor, self).__init__(receiver_id=receiver_id_1, monitor_timeout=monitor_timeout,
                                                         threshold=threshold)

        self.rx1 = StubPosMonitor(receiver_id=receiver_id_1, monitor_timeout=monitor_timeout, time_window=time_range)
        self.rx2 = StubPosMonitor(receiver_id=receiver_id_2, monitor_timeout=monitor_timeout, time_window=time_range)

        self._receivers = {
            receiver_id_1: self.rx1,
            receiver_id_2: self.rx2
        }

        # Set internal variable for minimum numbers of samples to average.
        self._min_samples = minimum_samples

        self._last_update = -float('inf')

        # Update the id string too
        self._id_str = '{} for {} and {}'.format(self.__class__.__name__, self.rx1.receiver_id, self.rx2.receiver_id)

    @property
    def time_range(self):
        """
        @brief The number of seconds worth of history to consider when computing the metric.

        This must be a positive number.
        """
        return self.rx1.samples.target_elapsed_time

    @time_range.setter
    def time_range(self, val):
        if val <= 0:
            raise ValueError('The time range for the dual distance monitor must be positive but got %s' % str(val))

        self.rx1.samples.target_elapsed_time = val
        self.rx2.samples.target_elapsed_time = val

    @property
    def required_num_samples(self):
        """
        @brief The minimum number of samples required to compute the average.

        This helps restrict the expected noise in the system and allows for a tighter threshold. A good number is 10
        samples.
        """
        return self._min_samples

    @monitor.Monitor.threshold.setter
    def threshold(self, val):
        if val <= 0:
            raise ValueError('Expected threshold to be positive; instead got %s' % str(val))

        self._threshold = val
        self._status['threshold'] = val

    def reset(self):
        super(DualAntennaDistanceMonitor, self).reset()

        self._last_update = None

        self.rx1.reset()
        self.rx2.reset()

    def _calculate_metric(self, message):
        """
        Updates the monitor with the latest sample and determine if an alarm should be raised.

        @param message The new sample.

        @return The new metric or None if the message is irrelevant or invalid.
        """

        time = message['rxTime']
        receiver = message['receiver_id']

        self._receivers[receiver].update(message)

        # Check if there was a new event for each receiver since the last update
        if not (self.rx1.samples.newest_time > self._last_update and
                self.rx2.samples.newest_time > self._last_update):
            return None

        # Make sure there are enough samples
        if len(self.rx1.samples) < self._min_samples or len(self.rx2.samples) < self._min_samples:
            self.logger.debug('At least one of the receivers has not seen enough samples yet: %s: %d; %s: %d',
                              self.rx1.receiver_id, len(self.rx1.samples), self.rx2.receiver_id, len(self.rx2.samples))
            return None

        # Keep track of the time of this update
        self._last_update = time

        # Compute the metric
        avg_rx1 = self.rx1.samples.sum / len(self.rx1.samples)
        avg_rx2 = self.rx2.samples.sum / len(self.rx2.samples)

        return math.sqrt(sum((avg_rx1 - avg_rx2) ** 2))

    def _compare_metric(self, metric):
        return metric <= self._threshold

    def verify_message(self, message):
        """
        Ensure that the message is from this monitor's receiver, is in-order (time-wise), and that the data source was
        valid.

        @param message: The message to verify.
        @return: True if the message is valid and false otherwise.
        """

        if 'rxTime' not in message or 'validity' not in message or 'receiver_id' not in message:
            self.logger.debug('Received an invalid message; missing basic fields rxTime, validity, or receiver_id')
            return False

        if not message['validity']:
            self.logger.debug('Received a message that flagged itself as invalid')
            return False

        if message['receiver_id'] not in self._receivers:
            self.logger.debug('Message is from a different receiver %s, instead of %s or %s',
                              message['receiver_id'],
                              self.rx1.receiver_id, self.rx2.receiver_id)

            return False

        if self._last_event_time is not None and message['rxTime'] < self._last_event_time:
            self.logger.warning('Received a message out of order. Time is %s but last time was %s',
                                message['rxTime'],
                                self._last_event_time)
            return False

        return True
