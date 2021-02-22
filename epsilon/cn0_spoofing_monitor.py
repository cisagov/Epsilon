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


class CnoSpoofingMonitor(monitor.Monitor):
    """
    @brief Implements a basic C/N0 spoofing monitor.

    This monitor tracks a single channel from a single receiver. If the C/N0 off of that signal is too high, then
    spoofing is declared.
    """

    def __init__(self, receiver_id, channel_id, threshold=51):
        """
        @brief Creates a new monitor.

        @param receiver_id The receiver ID to track signals from
        @param channel_id The C/N0 channel to monitor (in the form GnssId.SvId)
        @param threshold The C/N0 threshold to declare spoofing in dBHz.
        """

        super(CnoSpoofingMonitor, self).__init__(receiver_id=receiver_id, monitor_timeout=None, threshold=threshold)

        if '.' not in channel_id:
            raise ValueError('Channel id must be in the form GnssId.SvId but got %s' % channel_id)

        parts = channel_id.split('.')

        self._gnss_id = int(parts[0])
        self._sv_id = int(parts[1])

    def _calculate_metric(self, message):
        """
        @brief Processes the next message
        """

        # Determine if this instance should even process the event.
        if 'svs' not in message:
            return None

        for sv in message['svs']:
            if sv['gnssId'] == self._gnss_id and sv['svid'] == self._sv_id:
                return sv['cno']

        return None     # No matching channel
