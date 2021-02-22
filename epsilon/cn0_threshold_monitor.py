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

from epsilon import monitor


class CnoThresholdJammingMonitor(monitor.Monitor):
    """
    @brief Implements a basic C/N0 jamming monitor.

    This monitor checks whether all the C/N0 measurements in a time frame are below a threshold. If so, jamming is
    declared.
    """

    def __init__(self, receiver_id, threshold=20, time_window=5):
        """
        @brief Creates a new monitor.

        @param receiver_id The receiver ID to process events from.
        @param threshold The C/N0 threshold in dBHz
        @param time_window The amount of time for which a cno sample is valid
        """

        super(CnoThresholdJammingMonitor, self).__init__(receiver_id=receiver_id, monitor_timeout=None,
                                                         threshold=threshold)

        self._time_window = 0   # Initialize in init
        self.time_window = time_window  # Then set with error checking

        # Map gnss_id.sv_id to (time, cno) for each sv seen
        self._cnos = {}

    @property
    def time_window(self):
        """
        @brief The number of seconds a C/N0 sample is valid.

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
        super(CnoThresholdJammingMonitor, self).reset()
        self._cnos = {}

    def _calculate_metric(self, message):
        """
        @brief Processes CN0 messages (NOTE: each message is expected to contain an "svs" element that is a list of
               sv data entries, each entry containing "gnssId", "svid", "cno", and "qualityInd"
        """

        if 'svs' not in message:
            return None

        time = message['rxTime']

        for sv_data in message['svs']:
            if 'svid' not in sv_data or 'gnssId' not in sv_data or 'cno' not in sv_data:
                self.logger.error('Message with sv list is missing entires. Expected svid, gnssid, cno, and qualitInd'
                                  'but got %s', sv_data.keys())

            sv_id = sv_data['svid']
            gnss_id = sv_data['gnssId']
            cno = sv_data['cno']
            quality_ind = sv_data['qualityInd']

            if quality_ind < 4 or cno <= 0:
                self.logger.deubg('Ignoring cno value that is invalid. Qualit ind is %s and cno is %s',
                                  quality_ind, cno)
                continue

            channel_id = "{!s}.{!s}".format(gnss_id, sv_id)

            # Drop this message if it's out of order
            if channel_id in self._cnos:
                if self._cnos[channel_id] is not None and time < self._cnos[channel_id][0]:
                    self.logger.debug('Discarding time in the past; message time is %s but last time was %s', str(time),
                                      str(self._cnos[channel_id][0]))
                    continue

            # Update if the message is in order
            self._cnos[channel_id] = (time, cno)

        # After updating all of the cnos in this message, iterate over them, drop out-of-date values,
        #  and determine the max
        max_cno = None
        for key, (last_time, cno) in self._cnos.items():
            if time - last_time > self.time_window:
                self._cnos[key] = None
            elif max_cno is None or cno > max_cno:
                max_cno = cno

        return max_cno

    def _compare_metric(self, metric):
        return metric < self._threshold
