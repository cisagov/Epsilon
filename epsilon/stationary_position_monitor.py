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

import numpy as np

from epsilon import filtered_monitor


class StationaryPositionMonitor(filtered_monitor.FilteredMonitor):
    """
    @brief A class that monitors position solutions from a stationary receiver, flagging abnormal solutions.

    A stationary receiver theoretically has a constant position, but noise ensures that each PNT solution will deviate
    from this position slightly.  By tracking the average of the measurements, it is possible to identify solution
    outliers. These solution outliers may indicate an inauthentic signal is present in the navigation solution.

    If the monitor detects abnormal solutions, the @c integrity_status flag in the IntegrityStatusEvent output will
    become false, indicating that the integrity of the solution cannot be confirmed.

    This algorithm iteratively calculates the average of "good" measurements. If it exceeds a spoofing threshold, a
    spoofing flag is triggered.

    The spoofing flags enter an M-of-N detector to detect spoofing. If the spoofing flag was triggered for M of the last
    N measurements, the monitor will flag the solution as possibly invalid

    This monitor is a modified implementation of the position monitor in
    Dougherty, Ryan & Kurp, Timothy, _GPS Spoofing Detection for Stationary Resource-Constrained C/A Code Receivers_,
    MITRE MTR Draft, 8 October 2015.

    Testing revealed that the position data from a real receiver may not be normally distributed, making monitor
    initialization difficult.

    In this implementation, the covariance matrix was removed, which makes the units of the test statistic be meters
    offset from average, rather than a normalization factor. While decreasing sensitivity slightly, the metric is much
    more intuitive and easier to analyze and compute for canned scenarios.
    """

    def __init__(self, receiver_id, monitor_timeout=60, min_detections=3, sample_window=4, rejection_threshold=21.1,
                 spoofing_threshold=21.1, num_init_samples=30):
        """
        @brief Constructs a new StationaryPositionMonitor.

        @param receiver_id The ID of the receiver for this instance to monitor
        @param monitor_timeout The maximum number of seconds allowed between PNT events before resetting the monitor
        @param min_detections The number of detections to see before raising an alarm
        @param sample_window The number of most-recent samples to search for min_detections
        @param rejection_threshold The squared difference of position beyond which measurements will not be incorporated
               into the average
        @param spoofing_threshold The squared difference of position beyond which measurements will trigger a spoofing
               flag
        @param num_init_samples The number of samples with which to initialize
               the monitor statistic; these samples will be automatically accepted in the calculation
        """

        super(StationaryPositionMonitor, self).__init__(receiver_id=receiver_id, monitor_timeout=monitor_timeout,
                                                        threshold=spoofing_threshold, min_detections=min_detections,
                                                        sample_window=sample_window)

        if num_init_samples < 3:
            raise ValueError('The stationary position monitor must be initialized with at least 3 samples but the'
                             'number of initialization samples was set to %s' % str(num_init_samples))

        self._num_accepted = 0
        self._average = None    # Will be np.array([x,y,z])

        self._num_init_samples = num_init_samples

        # Declare the variables in __init__ but use the property setters to do error checking still
        self._rejection_threshold = None

        self.rejection_threshold = rejection_threshold

    @property
    def rejection_threshold(self):
        """
        @brief Sets the threshold for the offset test to reject a measurement.

        The offset test threshold to reject a measurement from the average calculations.

        The threshold must be positive.

        The rejection threshold should be less than or equal to the spoofing threshold, depending on how conservative
        the average calculation should be. a good choice is for it to be the same as the spoofing threshold.
        """
        return self._rejection_threshold

    @rejection_threshold.setter
    def rejection_threshold(self, rejection_threshold):
        if rejection_threshold < 0:
            self.logger.warning('Setting %s threshold to %f, the speed equivalent of the given %f velocity',
                                self._id_str, abs(rejection_threshold), rejection_threshold)

            rejection_threshold = abs(rejection_threshold)

        self._rejection_threshold = rejection_threshold

    @filtered_monitor.FilteredMonitor.threshold.setter
    def threshold(self, threshold):
        if threshold < 0:
            self.logger.warning('Setting %s threshold to %f, the speed equivalent of the given %f velocity',
                                self._id_str, abs(threshold), threshold)

            threshold = abs(threshold)

        self._threshold = threshold
        self._status['threshold'] = self._threshold

    @property
    def spoofing_threshold(self):
        """
        @brief Sets the threshold for the offset test to detect spoofing.

        The offset test threshold to reject a measurement from the average calculations.

        The threshold must be positive.

        The spoofing threshold is best determined empirically. First, collect a large amount of data (at least a day),
        and then choose the threshold to achieve a specified probability of false alarm.
        """
        return self._threshold

    @spoofing_threshold.setter
    def spoofing_threshold(self, spoofing_threshold):
        self.threshold = spoofing_threshold

    @property
    def average(self):
        """
        @brief Returns the average value of the filter.

        This can be set using the @ref hotStartMonitor method which allows the user to specify an average and number of
        samples accepted into that average.
        """
        return self._average

    @property
    def num_accepted(self):
        """
        @brief Returns the number of measurements used in the average calculation.

        This can be set using the @ref hotStartMonitor method which allows the user to specify an average and number of
        samples accepted into that average.
        """
        return self._num_accepted

    def _calculate_metric(self, message):
        """
        Updates the monitor with the latest sample and determine if an alarm should be raised.

        @param message The new sample.

        @return The new metric or None if the message is irrelevant or invalid.
        """

        if 'ecef_position' not in message:
            self.logger.debug('Got a message in stationary position monitor with no ecef_position field!')
            return None

        position = message['ecef_position']

        # Special case for first valid measurement.
        if self.num_accepted == 0:
            self.logger.debug('Accepted First measurement. [receiver_id = %s]', self.receiver_id)
            self._average = np.array(position)
            self._num_accepted = 1
            return None

        # Special case while initializing monitor (accept all samples, ignoring rejection_threshold)
        if self.num_accepted < self._num_init_samples:
            self.logger.debug(
                'Initializing statistics. [receiver_id = %s, current sample number = %s, number to initialize = %s]',
                self.receiver_id, self.num_accepted, self._num_init_samples
            )

            self._update_average(position)
            return None

        # Default case
        # No need to adjust coordinate frame for precision.
        shifted_position = position - self.average
        shifted_position_magnitude = np.linalg.norm(shifted_position)

        # Compare shifted_position against rejection threshold.
        # Do not use ecef_position in average if > rejection_threshold
        if shifted_position_magnitude < self.rejection_threshold:
            self._update_average(position)
        else:
            self.logger.info('Rejected position measurement: [shifted_pos magnitude = %s, Rejection Threshold = %s',
                             shifted_position_magnitude, self.rejection_threshold)

        # Compare shifted_position against spoofing_threshold regardless of whether it was factored into the average
        return shifted_position_magnitude

    def _update_average(self, measurement):
        """
        @brief Updates the average.

        @param measurement A vector holding the current receiver position in ECEF meters.
        """

        self._average = (self.average + (measurement - self.average) / (self.num_accepted + 1.0))
        self._num_accepted += 1

    def reset(self):
        """
        @brief Resets the entire monitor.
        """

        self.logger.debug('Fully resetting monitor [receiver_id = %s]',
                          self.receiver_id)

        super(StationaryPositionMonitor, self).reset()

        self._average = np.array([0, 0, 0])
        self._num_accepted = 0

    def hot_start_monitor(self, average, num_accepted):
        """
        @brief Initialize the monitor with an average and number of accepted samples.

        This method provides a mechanism to "hot-start" the monitor with data as though it had been running for hours or
        days.

        @param average The average ECEF position of the antenna in meters. This must be a 3 element iterable.

        @param num_accepted The number of accepted samples in the average calculation. This must be an integer.
        """

        # Perform error checking on the input parameters.
        if len(average) != 3:
            raise ValueError('Expected the average to have 3 components; instead got %d' % len(average))
        if not isinstance(num_accepted, int):
            raise ValueError('Expected the number of accepted samples to be an integer; instead got %s' %
                             str(num_accepted))

        if num_accepted <= 0:
            raise ValueError('Cannot have accepted fewer than 0 samples; got %d' % num_accepted)

        # Set the parameters.
        self._average = np.array(average)
        self._num_accepted = num_accepted
