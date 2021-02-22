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


class CnoDropJammingMonitor(monitor.Monitor):
    """
    @brief Implements a C/N0 drop monitor.

    The C/N0 drop monitor examines the C/N0 ratio on all available signals for a drop of some amount in a given time
    frame.

    The monitor requires at least two samples from each signal within the time window for this monitor to work. If two
    samples are not present, that signal will be ignored for the purposes of this monitor.
    """

    def __init__(self, receiver_id, threshold=-1, time_window=5):
        """
        @brief Creates a new monitor.

        @param receiver_id The receiver ID to track.
        @param threshold The C/N0 drop that needs to be seen by all signals within the time window.
        @param time_window The maximum number of seconds to look for the C/N0 drop over. Samples outside this time
               window will be discarded.
        """

        super(CnoDropJammingMonitor, self).__init__(receiver_id=receiver_id, monitor_timeout=None, threshold=threshold)

        self._time_window = 0
        self.time_window = time_window

        self._cnos = {}
        self._drops = {}    # Keep track of the drop data

    @property
    def time_window(self):
        """
        @brief The number of seconds a C/N0 sample is valid

        For each signal, a measurement will be discarded if either a new measurement comes in, or if the measurement is
        older than the time window.

        This must be non-negative.
        """

        return self._time_window

    @time_window.setter
    def time_window(self, val):
        if val < 0:
            raise ValueError('Expected time_window to be non-negative but got {}'.format(val))

        self._time_window = val

    def reset(self):
        super(CnoDropJammingMonitor, self).reset()

        self._cnos = {}
        self._drops = {}

    def _calculate_metric(self, message):
        """
        @brief Processes a new C/N0 message.
        """

        if 'svs' not in message:
            return None

        time = message['rxTime']

        for sv_data in message['svs']:
            if 'svid' not in sv_data or 'gnssId' not in sv_data or 'cno' not in sv_data or 'qualityInd' not in sv_data:
                self.logger.debug('Ignoring sv data entry that was missing a field. Expected svid, gnssId, cno, and'
                                  'qualityInd but got %s', sv_data.keys())
                continue

            sv_id = sv_data['svid']
            gnss_id = sv_data['gnssId']
            cno = sv_data['cno']
            quality_ind = sv_data['qualityInd']

            if quality_ind < 4 or cno <= 0:
                self.logger.debug('Ignoring cno value that is invalid. Qualit ind is %s and cno is %s',
                                  quality_ind, cno)
                continue

            channel_id = "{!s}.{!s}".format(gnss_id, sv_id)

            # If this channel is new, initialize its entry
            if channel_id not in self._cnos:
                self._cnos[channel_id] = buffers.TimedBuffer(self.time_window, keep_one_sample_before=False)

            self._cnos[channel_id].append(time, cno)

        alarm = True
        any_examined = False    # Make sure at least one SV was examined before returning the alarm value
        for channel, data in self._cnos.items():
            # Even if a new sample hasn't come in, make sure old ones are evicted (no need
            #  to provide the keep_one_before parameter here because it was set in for the
            #  TimedBuffer when it was created (above)
            data.remove_old_samples()

            if len(data) < 2:
                self._drops[channel] = None
                continue

            any_examined = True
            self._drops[channel] = -(data.newest_sample - data.oldest_sample)
            if self._drops[channel] <= self._threshold:
                alarm = False

        # If there was not enough data, return None
        if not any_examined:
            return None

        return alarm

    def _compare_metric(self, metric):
        """
        In this subclass, compute metric directly computes the alarm too.
        """

        return metric
