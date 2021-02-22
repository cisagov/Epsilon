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

from epsilon import filtered_monitor


class StationaryVelocityMonitor(filtered_monitor.FilteredMonitor):
    """
    @brief Monitors velocity solutions from a stationary receiver for abnormally large deviations.

    A stationary receiver theoretically has zero velocity, but noise ensures that each PNT solution will have some small
    nonzero velocity component.  This monitor examines the PNT solutions from a stationary receiver, looking for large
    velocity solutions, which may indicate an inauthentic signal is present in the navigation solution.

    This algorithm uses a simple threshold metric and an M-of-N detector to detect spoofing. If the receiver's absolute
    speed has exceeded the threshold in M of the last N samples, the solution is flagged as invalid

    Filtering detections by checking the last M of N samples helps to reduce false alarms from random noise spikes.

    This monitor is an implementation of the velocity monitor in
    Dougherty, Ryan & Kurp, Timothy, _GPS Spoofing Detection for Stationary Resource-Constrained C/A Code Receivers_,
    MITRE MTR Draft, 8 October 2015.
    """

    def __init__(self, receiver_id, monitor_timeout=60, min_detections=3, sample_window=4, threshold=0.5):
        """
        Create a new StationaryVelocityMonitor.

        @param receiver_id The ID of the receiver for this instance to monitor.
        @param monitor_timeout The maximum number of seconds allowed between PNT events before resetting the monitor
        @param min_detections The number of detections to see before raising an alarm
        @param sample_window The number of most-recent samples to search for min_detections
        @param threshold The squared speed threshold for the receiver in @f[m^2/s^2@f]. This must be positive
        """

        super(StationaryVelocityMonitor, self).__init__(receiver_id=receiver_id, monitor_timeout=monitor_timeout,
                                                        threshold=threshold, min_detections=min_detections,
                                                        sample_window=sample_window)

        self.logger.info('Created new StationaryVelocityMonitor with a threshold of %f to track receiver id %s',
                         self._threshold, self.receiver_id)

    @filtered_monitor.FilteredMonitor.threshold.setter
    def threshold(self, threshold):
        """
        Change the threshold; NOTE: this will log a warning if a negative value is given but will keep running
        using the absolute value

        @param threshold: The new threshold to use.
        """

        if threshold < 0:
            self.logger.warning('Setting %s threshold to %f, the speed equivalent of the given %f velocity',
                                self._id_str, abs(threshold), threshold)

            threshold = abs(threshold)

        self._threshold = threshold
        self._status['threshold'] = threshold

    def _calculate_metric(self, message):
        """
        Updates the monitor with the latest sample.

        @param message The new sample

        @return The updated metric or None if the message wasn't pertinent.
        """

        # Make sure the message is pertinent (right receiver, right fields, etc...)
        if 'ecef_velocity' not in message:
            self.logger.debug('Got a message with in stationary velocity monitor no ecef_velocity field!')
            return None

        velocity = message['ecef_velocity']

        if not hasattr(velocity, '__len__') or len(velocity) != 3:
            self.logger.error('Velocity field in message does not have 3 components; got %s', message)
            return None

        # If it's valid, calculate and return the metric
        return  sum([x**2 for x in message['ecef_velocity']])
