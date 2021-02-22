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


class CCDMonitor(monitor.Monitor):
    """
    @brief Looks for inconsistencies between the clock bias and clock rate in the PVT solution.

    This class monitors the clock bias and clock bias rate from the PVT solution. If the clock terms diverge, this
    monitor triggers a spoofing alarm.

    Under ideal conditions, the change in clock bias (cdr) over a period of time should be identical to the integral of
    the clock bias rate (cdr_dot) over that same period of time. In reality, factors such as environmental and
    measurement noise in the PVT solution yield small deviations between these two values, where the deviation is
    referred to as divergence.  Spoofing can induce very large divergence of these values if the spoofer cannot exactly
    compensate for the receiver position and velocity at all times, or if the spoofer is pulling away the timing
    solution.

    This algorithm continually computes clock consistency divergence (in meters) over a sliding window, and throws
    alarms when it has exceeded some threshold.

    This algorithm expects to receive the clock bias in terms of delta clock biases (i.e. the change from the previous
    sample). This limits the effect of unbiased clock drift as the change is only stored for a period of time.

    This monitor is an implementation of the CCD monitor in
    Dougherty, Ryan & Kurp, Timothy, _GPS Spoofing Detection for
    Stationary Resource-Constrained C/A Code Receivers_, MITRE MTR Draft, 8 October 2015.
    """

    def __init__(self, receiver_id, monitor_timeout=10.0, threshold=43.6, min_delta_t=30.0, max_delta_t=40.0):
        """
        @brief Creates a new CCD monitor.

        @param receiver_id The receiver this monitor should track.
        @param threshold The threshold for this monitor in meters equivalent divergence.
        @param min_delta_t The minimum number of samples (PNT events) over which to compute each CCD value. It must be
               positive and less than or equal to @c max_delta_t.
        @param max_delta_t The maximum number of samples (PNT events) over which to compute each CCD value. It must be
               positive and greater than or equal to @c min_delta_t.
        """

        super(CCDMonitor, self).__init__(receiver_id=receiver_id, monitor_timeout=monitor_timeout, threshold=threshold)

        if min_delta_t > max_delta_t:
            raise ValueError('Cannot have a minimum delta t greater than max delta t: got min %s and max %s' %
                             (str(min_delta_t), str(max_delta_t)))

        self.bias_samples = buffers.TimedRunningSum(target_elapsed_time=min_delta_t)

        # This holds the terms of the trapezoidal integral approximation:
        #  (new sample + last sample) / 2 * (new time - last time)
        self.rate_integral = buffers.TimedRunningSum(target_elapsed_time=min_delta_t)

        # Keep track of the last bias sample to figure out the new trapezoidal integral term when the next sample
        #  comes in (since the self.rate_integral buffer only holds integral terms, not samples)
        self._last_rate = None
        self._last_time = None

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
        return self.bias_samples.target_elapsed_time

    @min_delta_t.setter
    def min_delta_t(self, min_delta_t):
        if min_delta_t <= 0:
            raise ValueError('Minimum time must be positive! Instead got %s' % str(min_delta_t))

        self.bias_samples.target_elapsed_time = min_delta_t
        self.rate_integral.target_elapsed_time = min_delta_t

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
        super(CCDMonitor, self).reset()
        self.rate_integral.reset()
        self.bias_samples.reset()

        self._last_rate = None
        self._last_time = None

    def _calculate_metric(self, message):
        """
        @brief Computes the current test statistic for the monitor.

        The test statistic is the difference between the change in clock bias and the integral of the clock rate.

        Because the clock rate is the derivative of the clock bias, it's integral is simply the change in the clock
        bias.

        @param message The message to process
        @return The CCD test statistic in meters equivalent or None if the message could not be processed
        """

        if 'clock_rate' not in message or 'clock_bias' not in message:
            return None

        time = message['rxTime']

        clock_rate = message['clock_rate']
        clock_bias = message['clock_bias']

        # Need at least 2 bias samples to approximate the bias integral
        if self._last_rate is None:
            self._last_time = time
            self._last_rate = float(clock_rate)
            return None

        self.bias_samples.append(time, clock_bias)

        rate_term = (clock_rate + self._last_rate) / 2.
        rate_term *= (float(time) - self._last_time)
        self.rate_integral.append(time, rate_term)

        self._last_rate = float(clock_rate)
        self._last_time = time

        # One sample is kept beyond the min_delta_t window; make sure it doesn't exceed max_delta_t
        while self.bias_samples.elapsed_time > self._max_delta_t:
            # If it does, pop it
            self.bias_samples.popleft()
            self.rate_integral.popleft()

        # Ensure enough data has been saved to be able to compute results
        if self.bias_samples.elapsed_time < self.min_delta_t:
            self.logger.debug('Not enough saved filter output to process event. [Number of saved sampled = %s,'
                              'Elapsed time from oldest sample to current sample = %s seconds, Minimum elapsed time '
                              'required = %s seconds]',
                              len(self.bias_samples),
                              self.bias_samples.elapsed_time,
                              self.min_delta_t)
            return None

        # We want the change in clock bias from the first sample to the last
        # sample. If we have the delta clock bias between samples, it's just
        # summing everything except the first sample (which has the change from
        # the 0th sample to the first sample).
        x1 = self.bias_samples.sum
        x2 = self.rate_integral.sum

        return abs(x1 - x2)
