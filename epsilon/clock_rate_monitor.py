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

from epsilon import buffers, monitor


class ClockRateMonitor(monitor.Monitor):
    """
    @brief A class that monitors the change in the clock rate solution, flagging any that are outside the oscillator's
           normal operating bounds.

    This monitor relies on the fact that a receiver's oscillator is stable, meaning the clock rate solution should vary
    slowly with time.  Some drift is possible from changes in temperature and age, but these tend to be slow-acting.  If
    an attacker attempts to modify the clock rate solution outside the bounds of the normal drift, it can be detected.

    This monitor is an implementation of the clock rate monitor in Dougherty, Ryan & Kurp, Timothy, _GPS Spoofing
    Detection for Stationary Resource-Constrained C/A Code Receivers_, MITRE MTR Draft, 8 October 2015.
    """

    def __init__(self, receiver_id, monitor_timeout=10.0, threshold=0.0015, min_delta_t=60.0, max_delta_t=120.0):
        """
        @brief Constructs a new ClockRateMonitor.

        @param receiver_id The ID of the receiver to track with this monitor.
        @param monitor_timeout The maximum number of seconds between the oldest and newest samples in the FIR filter
               allowed before forcing the filter to reset.
        @param min_delta_t The minimum number of seconds required separating the two clock rate measurements used to
               compute the clock rate's rate of change. This must be positive.
        @param max_delta_t The maximum number of seconds required separating the two clock rate measurements used to
               compute the clock rate's rate of change. This must be positive.
        @param threshold The monitor's threshold for the clock rate's rate of change in m/s^2 equivalent.
        """

        super(ClockRateMonitor, self).__init__(receiver_id=receiver_id, monitor_timeout=monitor_timeout,
                                               threshold=threshold)

        if min_delta_t > max_delta_t:
            raise ValueError('Cannot have a minimum delta t greater than max delta t: got min %s and max %s' %
                             (str(min_delta_t), str(max_delta_t)))

        self.samples = buffers.TimedBuffer(target_elapsed_time=min_delta_t)

        self._max_delta_t = None
        self.max_delta_t = max_delta_t

    @monitor.Monitor.threshold.setter
    def threshold(self, threshold):
        if threshold <= 0:
            raise ValueError('Threshold must be positive! Instead got %s' % str(threshold))

        self._threshold = threshold
        self._status['threshold'] = threshold

    @property
    def min_delta_t(self):
        """
        @brief The minimum number of seconds over which to compute the clock rate change metric. This must be positive.

        In addition to being the minimum number of seconds to use to compute the metric, it is also the "default"
        value. The monitor chooses data to align as closely as possible with this value.

        As usual, there is a trade-off between choosing a small or large value for @min_delta_t. A small value can
        detect an attack faster than a larger value. A small value can also detect short attacks better than a larger
        value.

        Larger values are more sensitive than smaller values and do best detecting ramps.

        Of course, multiple instances of the monitor could be run, one with a shorter @c min_delta_t and another with a
        longer one. However, a good compromise is choosing @c min_delta_t to be 30 seconds.
        """
        return self.samples.target_elapsed_time

    @min_delta_t.setter
    def min_delta_t(self, min_delta_t):
        if min_delta_t <= 0:
            raise ValueError('Minimum time must be positive! Instead got %s' % str(min_delta_t))

        self.samples.target_elapsed_time = min_delta_t

    @property
    def max_delta_t(self):
        """
        @brief The minimum number of samples (PNT events) over which to compute each clock rate change value. This must be
               positive.

        This term provides the ability to continue operating the monitor, even after periods of short jamming. The
        closer it is to @c min_delta_t, the shorter period of jamming required to effectively reset the
        monitor. However, the metric distribution is not static across different values of @f[\Delta_t@f]. As
        @f[\Delta_t@f] increases, the noise on the metric decreases.

        This may seem like @c max_delta_t can be infinitely large, but at a certain point, the assumption that the
        oscillator has not drifted much in the analysis time window will be invalid. This condition should be avoided.

        A reasonably large value for @c max_delta_t can be chosen. A good value is 300 seconds.
        """

        return self._max_delta_t

    @max_delta_t.setter
    def max_delta_t(self, max_delta_t):
        if max_delta_t <= 0:
            raise ValueError('Maximum delta t must be positive; instead got %s' % str(max_delta_t))

        self._max_delta_t = max_delta_t

    def reset(self):
        super(ClockRateMonitor, self).reset()
        self.samples.reset()

    def _calculate_metric(self, message):
        """
        @brief Calculate the clock rate metric

        @param message The message to process
        @return The clock rate metric or None if the message could not be processed
        """

        if 'clock_rate' not in message:
            return None

        time = message['rxTime']
        clock_rate = message['clock_rate']

        self.samples.append(time, clock_rate)

        # One sample is kept beyond the min_delta_t window; make sure it doesn't exceed max_delta_t
        if self.samples.elapsed_time > self._max_delta_t:
            # If it does, pop it
            self.samples.popleft()

        # Ensure enough data has been saved to be able to compute results
        if self.samples.elapsed_time < self.min_delta_t:
            self.logger.debug('Not enough saved filter output to process event. [Number of saved sampled = %s,'
                              'Elapsed time from oldest sample to current sample = %s seconds, Minimum elapsed time '
                              'required = %s seconds]',
                              len(self.samples),
                              self.samples.elapsed_time,
                              self.min_delta_t)
            return None

        # If there is enough data, calculate the metric
        cdr_dot_change = (abs(self.samples.newest_sample - self.samples.oldest_sample) /
                          self.samples.elapsed_time)

        return cdr_dot_change
