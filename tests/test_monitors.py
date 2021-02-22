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

import copy
import json
import logging
import numpy as np
import unittest

from epsilon import monitor
from epsilon import filtered_monitor
from epsilon import ccd_monitor
from epsilon import cn0_drop_monitor
from epsilon import cn0_spoofing_monitor
from epsilon import cn0_threshold_monitor
from epsilon import clock_rate_monitor as crm
from epsilon import stationary_velocity_monitor as svm
from epsilon import stationary_position_monitor as spm
from epsilon import dual_antenna_distance_monitor as dadm

logging.basicConfig(level=logging.CRITICAL)

TEST_RX_STR = 'Test Receiver'

# Minimum valid message; others will be deep copies this and add/change values as needed
BASIC_MESSAGE = {'receiver_id': TEST_RX_STR, 'rxTime': 1, 'validity': True}


class ThinMonitor(monitor.Monitor):
    """
    Thin subclass for Monitor so the abstract functionality can be tested
    """

    def __init__(self, receiver_id=None, monitor_timeout=None, threshold=0.):
        super(ThinMonitor, self).__init__(receiver_id=receiver_id, monitor_timeout=monitor_timeout, threshold=threshold)

    def _calculate_metric(self, message):
        return 0


class ThinFilteredMonitor(filtered_monitor.FilteredMonitor):
    """
    Thin subclass for FilteredMonitor to test
    """

    def __init__(self, receiver_id, monitor_timeout=None, threshold=0., min_detections=1, sample_window=1):
        super(ThinFilteredMonitor, self).__init__(receiver_id=receiver_id, monitor_timeout=monitor_timeout,
                                                  threshold=threshold, min_detections=min_detections,
                                                  sample_window=sample_window)

        # This will be used by some tests that don't care about time
        self.cur_time = 1

    def _calculate_metric(self, message):
        """
        Pass {'newMetric': True} to simulate an alert
        """

        self.cur_time += 1
        return message.get('newMetric', 0)


class TestMonitor(unittest.TestCase):
    """
    Tests for the monitor base class.
    """

    def test_empty_init(self):
        tester = ThinMonitor()

        self.assertIsNone(tester.receiver_id)
        self.assertIsNone(tester.monitor_timeout)
        self.assertIsNone(tester._last_event_time)
        self.assertDictEqual(tester._status, {'alarm': False, 'threshold': tester._threshold, 'metric': None})

    def test_arg_init(self):
        tester = ThinMonitor(receiver_id=TEST_RX_STR, monitor_timeout=2.1, threshold=1.2)

        self.assertEqual(tester.receiver_id, TEST_RX_STR)
        self.assertAlmostEqual(tester.monitor_timeout, 2.1)
        self.assertAlmostEqual(tester._threshold, 1.2)
        self.assertIsNone(tester._last_event_time)
        self.assertDictEqual(tester._status, {'alarm': False, 'threshold': tester._threshold, 'metric': None})

    def test_verify_message(self):
        tester = ThinMonitor(receiver_id=TEST_RX_STR, monitor_timeout=2.1)

        self.assertTrue(tester.verify_message({'receiver_id': TEST_RX_STR, 'rxTime': 123, 'validity': True}))
        self.assertFalse(tester.verify_message({'receiver_id': TEST_RX_STR, 'rxTime': 123, 'validity': False}))
        self.assertFalse(tester.verify_message({'receiver_id': 'Some other receiver', 'rxTime': 123, 'validity': True}))

        # Test an out of order message without using other code
        tester._last_event_time = 1234
        self.assertFalse(tester.verify_message({'receive_id': TEST_RX_STR, 'rxTime': 123, 'validity': True}))

    def test_get_status(self):
        tester = ThinMonitor()
        self.assertDictEqual(json.loads(tester.get_status()),
                             {'alarm': False, 'threshold': tester._threshold, 'metric': None})

    def test_create_from_config(self):
        # Use the thin wrapper, so this test configuration will only work in this test environment
        config = {
            'receiver_id': TEST_RX_STR,
            'monitor_timeout': 3,
            'threshold': 2
        }

        tester = monitor.from_config(monitor_name='ThinMonitor', configuration=config)

        self.assertIsInstance(tester, ThinMonitor)
        self.assertEqual(tester.receiver_id, TEST_RX_STR)
        self.assertEqual(tester.monitor_timeout, 3)
        self.assertEqual(tester._threshold, 2)

    def test_from_config_invalid(self):
        config = {
            'receive_id': TEST_RX_STR,
            'monitor_timeout': 3
        }

        self.assertIsNone(monitor.from_config('Nonexistent', config))

    def test_reset(self):
        tester = ThinMonitor()
        tester._last_event_time = 18
        tester._status['metric'] = 88

        self.assertEqual(tester._last_event_time, 18)

        tester.reset()

        self.assertIsNone(tester._last_event_time)
        self.assertDictEqual(tester._status, {'alarm': False, 'threshold': tester._threshold, 'metric': None})

    def test_metric_property(self):
        tester = ThinMonitor(receiver_id=TEST_RX_STR, monitor_timeout=3, threshold=0)

        self.assertFalse(tester.alarm)

        tester._status['alarm'] = True

        self.assertTrue(tester.alarm)

    def test_alarm_property(self):
        tester = ThinMonitor(receiver_id=TEST_RX_STR, monitor_timeout=3, threshold=0)

        self.assertIsNone(tester.metric)

        tester._status['metric'] = 3.14

        self.assertAlmostEqual(tester.metric, 3.14)

    def test_compare_metric(self):
        tester = ThinMonitor(receiver_id=TEST_RX_STR, monitor_timeout=3, threshold=1.2)

        self.assertFalse(tester._compare_metric(0.3))
        self.assertTrue(tester._compare_metric(37.4))

    def test_update(self):
        tester = ThinMonitor(receiver_id=TEST_RX_STR, monitor_timeout=3, threshold=0)

        # Test a valid message
        test_message = {
            'receiver_id': TEST_RX_STR,
            'rxTime': 4321,
            'validity': True
        }

        self.assertFalse(tester.update(test_message))

        # Check internal settings after the update
        self.assertEqual(tester._last_event_time, 4321)
        self.assertEqual(tester.metric, 0)

        # Test an invalid message
        test_message['validity'] = False
        self.assertIsNone(tester.update(test_message))
        self.assertEqual(tester._last_event_time, 4321)     # Should not change if message is not processed

        # Test an out-of-order message
        test_message['validity'] = True
        test_message['rxTime'] = 321
        self.assertIsNone(tester.update(test_message))

        # Test a message for another receiver
        test_message['receiver_id'] = 'Something else'
        self.assertIsNone(tester.update(test_message))

        # Test another valid message and make sure last time updates
        test_message['rxTime'] = 4323
        test_message['receiver_id'] = TEST_RX_STR
        self.assertFalse(tester.update(test_message))

        self.assertEqual(tester._last_event_time, 4323)

        # Test a timeout
        test_message['rxTime'] = 6000
        self.assertFalse(tester.update(test_message))
        self.assertEqual(tester._last_event_time, 6000)     # This should still update


class TestFilteredMonitor(unittest.TestCase):
    def test_create(self):
        tester = ThinFilteredMonitor(receiver_id=TEST_RX_STR)

        self.assertEqual(tester.receiver_id, TEST_RX_STR)
        self.assertIsNone(tester.monitor_timeout)
        self.assertAlmostEqual(tester._threshold, 0.)
        self.assertEqual(tester._min_detections, 1)
        self.assertEqual(tester.detections.maxlen, 1)
        self.assertEqual(len(tester.detections), 0)
        self.assertFalse(tester._status['alarm'])
        self.assertIsNone(tester.metric)
        self.assertFalse(tester._status['spoofing_flag'])

        tester = ThinFilteredMonitor(receiver_id=TEST_RX_STR, monitor_timeout=2, threshold=1.2,
                                     min_detections=3, sample_window=8)

        self.assertEqual(tester.receiver_id, TEST_RX_STR)
        self.assertEqual(tester.monitor_timeout, 2)
        self.assertAlmostEqual(tester._threshold, 1.2)
        self.assertEqual(tester._min_detections, 3)
        self.assertEqual(tester.detections.maxlen, 8)
        self.assertEqual(len(tester.detections), 0)

    def test_from_config(self):
        config = {
            'receiver_id': TEST_RX_STR,
            'monitor_timeout': 7.2,
            'min_detections': 4,
            'sample_window': 7,
            'threshold': 2.3
        }

        tester = monitor.from_config('ThinFilteredMonitor', config)

        self.assertIsNotNone(tester)
        self.assertIsInstance(tester, ThinFilteredMonitor)

        self.assertEqual(tester.receiver_id, TEST_RX_STR)
        self.assertEqual(tester.monitor_timeout, 7.2)
        self.assertAlmostEqual(tester._threshold, 2.3)
        self.assertEqual(tester._min_detections, 4)
        self.assertEqual(tester.detections.maxlen, 7)
        self.assertEqual(len(tester.detections), 0)

    def test_reset(self):
        tester = ThinFilteredMonitor(receiver_id='Tester')
        tester.detections.append(8)

        self.assertEqual(len(tester.detections), 1)
        self.assertEqual(tester.detections.sum, 8)

        tester.reset()

        self.assertEqual(len(tester.detections), 0)
        self.assertEqual(tester.detections.sum, 0)
        self.assertFalse(tester._status['alarm'])
        self.assertIsNone(tester.metric)
        self.assertFalse(tester._status['spoofing_flag'])

    def test_update(self):
        tester = ThinFilteredMonitor(receiver_id=TEST_RX_STR, monitor_timeout=10, threshold=0.,
                                     min_detections=3, sample_window=9)

        self.assertEqual(tester._min_detections, 3)

        ok_message = {'receiver_id': TEST_RX_STR, 'rxTime': 1, 'validity': True, 'newMetric': 0}
        bad_message = {'receiver_id': TEST_RX_STR, 'rxTime': 1, 'validity': True, 'newMetric': 1}

        # Test a bunch of messages that are acceptable
        for _ in range(9):
            self.assertFalse(tester.update(ok_message))

        self.assertEqual(len(tester.detections), 9)
        self.assertEqual(tester.detections.sum, 0)
        self.assertDictEqual(json.loads(tester.get_status()), {'alarm': False, 'spoofing_flag': False,
                                                               'metric': 0, 'threshold': 0.})

        # Start throwing in alarms (sample alarms--the monitor it self should not trigger yet)
        bad_message['rxTime'] = tester.cur_time
        self.assertFalse(tester.update(bad_message))
        bad_message['rxTime'] += 1

        self.assertEqual(len(tester.detections), 9)
        self.assertEqual(tester.detections.sum, 1)

        # Put in enough bad messages to trigger
        self.assertFalse(tester.update(bad_message))

        self.assertFalse(tester.update(ok_message))
        ok_message['rxTime'] = tester.cur_time

        self.assertTrue(tester.update(bad_message))    # First one that should trigger
        bad_message['rxTime'] = tester.cur_time

        self.assertEqual(tester.detections.sum, 3)
        self.assertDictEqual(json.loads(tester.get_status()), {'alarm': True, 'spoofing_flag': True,
                                                               'metric': 1, 'threshold': 0.})

        self.assertEqual(tester.metric, 1)

        # Fill the queue with bad messages
        for _ in range(9):
            self.assertTrue(tester.update(bad_message))

        self.assertEqual(tester.detections.sum, 9)
        self.assertEqual(len(tester.detections), 9)


class TestStationaryVelocityMonitor(unittest.TestCase):
    def test_create(self):
        tester = svm.StationaryVelocityMonitor(receiver_id=TEST_RX_STR)

        self.assertEqual(tester.receiver_id, TEST_RX_STR)
        self.assertEqual(tester.monitor_timeout, 60)
        self.assertAlmostEqual(tester._threshold, 0.5)
        self.assertEqual(tester._min_detections, 3)
        self.assertAlmostEqual(tester.threshold, 0.5, places=1)
        self.assertEqual(tester.detections.maxlen, 4)
        self.assertEqual(len(tester.detections), 0)

        tester = svm.StationaryVelocityMonitor(receiver_id=TEST_RX_STR, monitor_timeout=2,
                                               min_detections=3, sample_window=8, threshold=3.14)

        self.assertEqual(tester.receiver_id, TEST_RX_STR)
        self.assertEqual(tester.monitor_timeout, 2)
        self.assertEqual(tester._min_detections, 3)
        self.assertAlmostEqual(tester.threshold, 3.14, places=2)
        self.assertEqual(tester.detections.maxlen, 8)
        self.assertEqual(len(tester.detections), 0)

    def test_from_config(self):
        config = {
            'receiver_id': TEST_RX_STR,
            "threshold": 0.5,
            "min_detections": 3,
            "sample_window": 4,
            "monitor_timeout": 60
        }

        tester = monitor.from_config('StationaryVelocityMonitor', config)

        self.assertEqual(tester.receiver_id, TEST_RX_STR)
        self.assertEqual(tester._min_detections, 3)
        self.assertEqual(tester.detections.maxlen, 4)
        self.assertAlmostEqual(tester.threshold, 0.5)
        self.assertEqual(tester.monitor_timeout, 60)

    def test_threshold_set(self):
        tester = svm.StationaryVelocityMonitor(receiver_id='Test')
        tester.threshold = 2
        self.assertEqual(tester.threshold, 2)

        tester.threshold = 1
        self.assertEqual(tester.threshold, 1)

        # Negative inputs should turn to positive
        tester.threshold = -3
        self.assertEqual(tester.threshold, 3)

    def test_no_velocity_data(self):
        # Make sure a message with no velocity information is ignored
        tester = svm.StationaryVelocityMonitor(TEST_RX_STR)

        message = {'receiver_id': TEST_RX_STR, 'rxTime': 1, 'validity': True}
        self.assertIsNone(tester.update(message))

    def test_process_message(self):
        # Ensure the metric is computed correctly for a few different velocities
        tester = svm.StationaryVelocityMonitor(TEST_RX_STR)

        message = {'receiver_id': TEST_RX_STR, 'rxTime': 1, 'validity': True, 'ecef_velocity': [1.1, 2, 0.3]}
        self.assertAlmostEqual(tester._calculate_metric(message), 1.1 * 1.1 + 2 * 2 + 0.3 * 0.3, places=5)

        message = {'receiver_id': TEST_RX_STR, 'rxTime': 2, 'validity': True, 'ecef_velocity': [0, 0.1, 0.3]}
        self.assertAlmostEqual(tester._calculate_metric(message), 0 + 0.1 * 0.1 + 0.3 * 0.3, places=5)

        message = {'receiver_id': TEST_RX_STR, 'rxTime': 3, 'validity': True,
                   'ecef_velocity': [-1.1, 0.001, -0.2]}
        self.assertAlmostEqual(tester._calculate_metric(message), -1.1 * -1.1 + 0.001 * 0.001 + -0.2 * -0.2, places=5)

    def test_update_ok(self):
        tester = svm.StationaryVelocityMonitor(receiver_id=TEST_RX_STR)

        message = copy.deepcopy(BASIC_MESSAGE)
        message['ecef_velocity'] = (0, 0, 0)

        self.assertFalse(tester.update(message))
        self.assertEqual(tester.metric, 0)
        self.assertFalse(tester._status['alarm'])
        self.assertEqual(tester.detections.sum, 0)
        self.assertEqual(len(tester.detections), 1)

        message['ecef_velocity'] = (0.1, 0.2, 0.11)
        message['rxTime'] += 1

        self.assertFalse(tester.update(message))
        self.assertAlmostEqual(tester.metric, 0.1 * 0.1 + 0.2 * 0.2 + 0.11 * 0.11, places=5)
        self.assertFalse(tester._status['alarm'])
        self.assertEqual(tester.detections.sum, 0)
        self.assertEqual(len(tester.detections), 2)

        message['ecef_velocity'] = (0.1, 0.2, 0.11)
        message['rxTime'] += 1

        self.assertFalse(tester.update(message))
        self.assertAlmostEqual(tester.metric, 0.1 * 0.1 + 0.2 * 0.2 + 0.11 * 0.11, places=5)
        self.assertFalse(tester._status['alarm'])
        self.assertEqual(tester.detections.sum, 0)
        self.assertEqual(len(tester.detections), 3)

        message['ecef_velocity'] = (0., 0.1, 0.0)
        message['rxTime'] += 1

        self.assertFalse(tester.update(message))
        self.assertAlmostEqual(tester.metric, 0. + 0.1 * 0.1 + 0.0 * 0.0, places=5)
        self.assertFalse(tester._status['alarm'])
        self.assertEqual(tester.detections.sum, 0)
        self.assertEqual(len(tester.detections), 4)

        # Add another to fully test the deque's max length
        message['rxTime'] += 1

        self.assertFalse(tester.update(message))
        self.assertAlmostEqual(tester.metric, 0. + 0.1 * 0.1 + 0.0 * 0.0, places=5)
        self.assertFalse(tester._status['alarm'])
        self.assertEqual(tester.detections.sum, 0)
        self.assertEqual(len(tester.detections), 4)

    def test_update_bad(self):
        # Send in bad data
        tester = svm.StationaryVelocityMonitor(receiver_id=TEST_RX_STR)

        # Start with one good message
        message = copy.deepcopy(BASIC_MESSAGE)
        message['ecef_velocity'] = (0, 0, 0)

        self.assertFalse(tester.update(message))
        self.assertEqual(tester.metric, 0)
        self.assertFalse(tester._status['alarm'])
        self.assertEqual(tester.detections.sum, 0)
        self.assertEqual(len(tester.detections), 1)

        # Now pass the threshold
        message['ecef_velocity'] = (1, 2.1, 3)
        message['rxTime'] += 1

        self.assertFalse(tester.update(message))
        self.assertAlmostEqual(tester.metric, 1 * 1 + 2.1 * 2.1 + 3 * 3, places=5)
        self.assertFalse(tester._status['alarm'])
        self.assertEqual(tester.detections.sum, 1)      # This should be the only indication for now
        self.assertEqual(len(tester.detections), 2)

        message['rxTime'] += 1

        self.assertFalse(tester.update(message))
        self.assertAlmostEqual(tester.metric, 1 * 1 + 2.1 * 2.1 + 3 * 3, places=5)
        self.assertFalse(tester._status['alarm'])
        self.assertEqual(tester.detections.sum, 2)      # Again this is the only indication
        self.assertEqual(len(tester.detections), 3)

        message['ecef_velocity'] = (8, 1.3, 2.2)
        message['rxTime'] += 1

        self.assertTrue(tester.update(message))     # Now this should trip
        self.assertAlmostEqual(tester.metric, 8 * 8 + 1.3 * 1.3 + 2.2 * 2.2, places=5)
        self.assertTrue(tester._status['alarm'])        # Now this should have changed
        self.assertEqual(tester.detections.sum, 3)
        self.assertEqual(len(tester.detections), 4)

        # Add another good one (there should still be an alarm though)
        message['ecef_velocity'] = (0., 0.1, 0.0)
        message['rxTime'] += 1

        self.assertTrue(tester.update(message))
        self.assertAlmostEqual(tester.metric, 0. + 0.1 * 0.1 + 0.0 * 0.0, places=5)
        self.assertTrue(tester._status['alarm'])
        self.assertEqual(tester.detections.sum, 3)      # This should not have changed
        self.assertEqual(len(tester.detections), 4)     # Nor should this

    def test_out_of_order(self):
        tester = svm.StationaryVelocityMonitor(receiver_id=TEST_RX_STR)

        # Send a good message (rxTime is 1)
        message = copy.deepcopy(BASIC_MESSAGE)
        message['ecef_velocity'] = (0, 0, 0)

        self.assertFalse(tester.update(message))
        self.assertEqual(tester._last_event_time, 1)
        self.assertEqual(tester.metric, 0)
        self.assertFalse(tester._status['alarm'])
        self.assertEqual(tester.detections.sum, 0)
        self.assertEqual(len(tester.detections), 1)

        message['ecef_velocity'] = (0.1, 0.2, 0.11)
        message['rxTime'] = 0   # This is "in the past"

        self.assertIsNone(tester.update(message))
        self.assertEqual(tester.metric, 0)
        self.assertFalse(tester._status['alarm'])
        self.assertEqual(tester.detections.sum, 0)
        self.assertEqual(len(tester.detections), 1)     # This should not have changed


class TestStationaryPositionMonitor(unittest.TestCase):
    def testInitialize(self):
        tester = spm.StationaryPositionMonitor(receiver_id='Test Rx',
                                               rejection_threshold=3,
                                               spoofing_threshold=5,
                                               min_detections=3,
                                               sample_window=4,
                                               monitor_timeout=60,
                                               num_init_samples=4)

        # Only test parameters specific to this class--the others have been covered
        self.assertEqual(tester._rejection_threshold, 3)
        self.assertEqual(tester._threshold, 5)
        self.assertEqual(tester._num_init_samples, 4)

        self.assertEqual(tester._num_accepted, 0)
        self.assertIsNone(tester._average)

    def test_from_config(self):
        config = {
            'receiver_id': TEST_RX_STR,
            "rejection_threshold": 19.9,
            "spoofing_threshold": 19.0,
            "min_detections": 4,
            "sample_window": 7,
            "monitor_timeout": 12,
            'num_init_samples': 22
        }

        tester = monitor.from_config('StationaryPositionMonitor', config)

        self.assertEqual(tester.receiver_id, TEST_RX_STR)
        self.assertEqual(tester._min_detections, 4)
        self.assertEqual(tester.detections.maxlen, 7)
        self.assertEqual(tester.rejection_threshold, 19.9)
        self.assertEqual(tester.spoofing_threshold, 19.0)
        self.assertEqual(tester.monitor_timeout, 12)
        self.assertEqual(tester._num_init_samples, 22)

        self.assertEqual(tester._num_accepted, 0)
        self.assertIsNone(tester._average)

    def testRejectionThresholdSetter(self):
        tester = spm.StationaryPositionMonitor(receiver_id=TEST_RX_STR)

        tester.rejection_threshold = 2
        self.assertEqual(tester.rejection_threshold, 2)

        tester.rejection_threshold = 1
        self.assertEqual(tester.rejection_threshold, 1)

        # Negative thresholds should be positive
        tester.rejection_threshold = -2
        self.assertEqual(tester.rejection_threshold, 2)

    def testSpoofingThresholdSetter(self):
        tester = spm.StationaryPositionMonitor(receiver_id=TEST_RX_STR)

        tester.spoofing_threshold = 2
        self.assertEqual(tester.spoofing_threshold, 2)

        tester.spoofing_threshold = 1
        self.assertEqual(tester.spoofing_threshold, 1)

        # Negative thresholds should be positive
        tester.spoofing_threshold = -2
        self.assertEqual(tester.spoofing_threshold, 2)

    def testNoPositionData(self):
        # Ensure that messages without position data are ignored
        tester = spm.StationaryPositionMonitor(receiver_id=TEST_RX_STR)
        message = BASIC_MESSAGE

        self.assertIsNone(tester._average)
        self.assertEqual(tester._num_accepted, 0)

        self.assertIsNone(tester.update(message))

        # These should be unchanged
        self.assertIsNone(tester._average)
        self.assertEqual(tester._num_accepted, 0)

    def testAverage(self):
        pos_data = [[1, 2, 3], [0, 0, 0], [8, 10, 12], [3, 4, 5]]
        solutions = [[1, 2, 3], [0.5, 1, 1.5], [3, 4, 5], [3, 4, 5]]

        tester = spm.StationaryPositionMonitor(receiver_id=TEST_RX_STR)
        tester._average = np.array([0, 0, 0])
        for pos, sol in zip(pos_data, solutions):
            tester._update_average(pos)
            np.testing.assert_array_equal(tester._average, sol)

    def testMetricCalculation(self):
        # Ensure the metric is computed correctly for a few different
        # positions
        pos_data = [[0, 0, 0], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        metric = [None, None, None, None, np.sqrt(.5 ** 2 + .5 ** 2 + .5 ** 2)]

        tester = spm.StationaryPositionMonitor(receiver_id=TEST_RX_STR, num_init_samples=4)

        message = copy.deepcopy(BASIC_MESSAGE)
        for ind, data in enumerate(pos_data):
            message['ecef_position'] = data

            if metric[ind] is None:
                self.assertIsNone(tester._calculate_metric(message))
            else:
                self.assertEqual(tester._calculate_metric(message), metric[ind])

            message['rxTime'] += 1

    def testAlarm(self):
        pos_data = [[0, 0, 0], [2, 2, 2], [2, 2, 2], [2, 2, 2], [20, 20, 20],
                    [100, 100, 100], [500, 500, 500], [1000, 1000, 1000],
                    [1000, 1000, 1000]]

        tester = spm.StationaryPositionMonitor(receiver_id=TEST_RX_STR, rejection_threshold=3,
                                               spoofing_threshold=5,
                                               min_detections=3,
                                               sample_window=4,
                                               monitor_timeout=60,
                                               num_init_samples=4)

        results = [None, None, None, None, False, False, True, True]
        alarms = [False, False, False, False, False, False, True, True]
        spoofing_flags = [False, False, False, False, True, True, True, True]

        message = copy.deepcopy(BASIC_MESSAGE)
        for ind, (pos, res) in enumerate(zip(pos_data, results)):
            message['ecef_position'] = pos

            if res is None:
                self.assertIsNone(tester.update(message))
            else:
                self.assertEqual(tester.update(message), res)

            self.assertEqual(tester._status['alarm'], alarms[ind])
            self.assertEqual(tester._status['spoofing_flag'], spoofing_flags[ind])
            message['rxTime'] += 1

    def testHotStartMonitor(self):
        tester = spm.StationaryPositionMonitor(receiver_id=TEST_RX_STR)

        self.assertIsNone(tester._average)
        self.assertEqual(tester.num_accepted, 0)

        tester.hot_start_monitor([1, 2, 3], 40)
        np.testing.assert_array_equal(tester.average, [1, 2, 3])
        self.assertEqual(tester.num_accepted, 40)

    def testResetMonitor(self):
        tester = spm.StationaryPositionMonitor(receiver_id=TEST_RX_STR)
        tester.hot_start_monitor([1, 2, 3], 100)
        tester._status['metric'] = 8
        tester._status['alarm'] = True

        tester.reset()

        self.assertEqual(tester.num_accepted, 0)
        np.testing.assert_array_equal(tester.average, [0, 0, 0])
        self.assertIsNone(tester.metric)
        self.assertFalse(tester._status['alarm'])


class TestClockRateMonitor(unittest.TestCase):
    def test_create(self):
        tester = crm.ClockRateMonitor(receiver_id=TEST_RX_STR)

        self.assertEqual(tester.receiver_id, TEST_RX_STR)
        self.assertEqual(tester.monitor_timeout, 10)
        self.assertAlmostEqual(tester._threshold, 0.0015)

        tester = crm.ClockRateMonitor(receiver_id=TEST_RX_STR, monitor_timeout=2, threshold=3.14)

        self.assertEqual(tester.receiver_id, TEST_RX_STR)
        self.assertEqual(tester.monitor_timeout, 2)
        self.assertAlmostEqual(tester.threshold, 3.14, places=2)

    def test_from_config(self):
        config = {
            'receiver_id': TEST_RX_STR,
            "threshold": 0.5,
            "monitor_timeout": 60
        }

        tester = monitor.from_config('ClockRateMonitor', config)

        self.assertEqual(tester.receiver_id, TEST_RX_STR)
        self.assertAlmostEqual(tester.threshold, 0.5)
        self.assertEqual(tester.monitor_timeout, 60)

    def testMinDeltaTSetter(self):
        tester = crm.ClockRateMonitor(TEST_RX_STR)

        tester.min_delta_t = 120.0
        self.assertAlmostEqual(tester.min_delta_t, 120.0)

        tester.min_delta_t = 60.0
        self.assertAlmostEqual(tester.min_delta_t, 60.0)

        tester.min_delta_t = 5.5
        self.assertAlmostEqual(tester.min_delta_t, 5.5)

        # Negative should raise
        with self.assertRaises(ValueError):
            tester.min_delta_t = -2

    def testMaxDeltaTSetter(self):
        tester = crm.ClockRateMonitor(TEST_RX_STR)

        tester.max_delta_t = 60.0
        self.assertAlmostEqual(tester.max_delta_t, 60.0)

        tester.max_delta_t = 120.0
        self.assertAlmostEqual(tester.max_delta_t, 120.0)

        tester.max_delta_t = 5.5
        self.assertAlmostEqual(tester.max_delta_t, 5.5)

        # Invalid number should raise
        with self.assertRaises(ValueError):
            tester.max_delta_t = -2

    def testThresholdSetter(self):
        tester = crm.ClockRateMonitor(TEST_RX_STR)

        tester.threshold = 2
        self.assertEqual(tester.threshold, 2)

        tester.threshold = 0.0015
        self.assertAlmostEqual(tester.threshold, 0.0015)

        # Invalid number should raise
        with self.assertRaises(ValueError):
            tester.threshold = -2

    def testClockRateOutput1(self):
        tester = crm.ClockRateMonitor(TEST_RX_STR, threshold=1, min_delta_t=5, max_delta_t=5.1)

        message = copy.deepcopy(BASIC_MESSAGE)

        times = range(1, 11)
        results = [None, None, None, None, None, True, True, True, True, True]
        metrics = [None, None, None, None, None, 7.0, 9.0, 11.0, 13.0, 15.0]

        for (time, res, metric) in zip(times, results, metrics):
            message['rxTime'] = time
            message['clock_rate'] = time ** 2

            if res is None:
                self.assertIsNone(tester.update(message))
                self.assertFalse(tester._status['alarm'])
            else:
                self.assertEqual(tester.update(message), res)
                self.assertAlmostEqual(tester.metric, metric)
                self.assertEqual(tester._status['alarm'], res)

    def testClockRateOutput2(self):
        tester = crm.ClockRateMonitor(receiver_id=TEST_RX_STR, threshold=1, min_delta_t=5, max_delta_t=6.1)

        times = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                          9.0, 10.0, 11.0, 12.0, 13.0, 14.0])

        cdr_dot_data = times ** 2

        metrics = [None, None, None, None, None, 7.0, 12.0, 14.0, 16.0, 18.0, None, 23.0]
        results = [None, None, None, None, None, True, True, True, True, True, None, True]
        alarms = [False, False, False, False, False, True, True, True, True, True, True, True]

        message = copy.deepcopy(BASIC_MESSAGE)

        for (time, rate, res, metric, alarm) in zip(times, cdr_dot_data, results, metrics, alarms):
            message['rxTime'] = time
            message['clock_rate'] = rate

            if res is None:
                self.assertIsNone(tester.update(message))
            else:
                self.assertEqual(tester.update(message), res)
                self.assertAlmostEqual(tester.metric, metric)

            self.assertEqual(tester._status['alarm'], alarm)


class TestCCDMonitor(unittest.TestCase):
    def setUp(self):
        with open('ccd_monitor_test_vectors.json', 'r') as f:
            self._test_data = json.load(f)

    def get_test_data(self, name):
        if name not in self._test_data:
            raise KeyError('Invalid key for test case data: %s' % name)

        for item in self._test_data[name]:
            # Time, bias, rate
            yield (item['time'], item['cdr'], item['cdr_dot'])

    def test_create(self):
        tester = ccd_monitor.CCDMonitor(receiver_id=TEST_RX_STR)

        self.assertEqual(tester.receiver_id, TEST_RX_STR)
        self.assertEqual(tester.monitor_timeout, 10)
        self.assertAlmostEqual(tester._threshold, 43.6)
        self.assertAlmostEqual(tester.bias_samples._target_elapsed_time, 30.0)
        self.assertAlmostEqual(tester.max_delta_t, 40.0)

        tester = ccd_monitor.CCDMonitor(receiver_id=TEST_RX_STR, monitor_timeout=2, threshold=3.14)

        self.assertEqual(tester.receiver_id, TEST_RX_STR)
        self.assertEqual(tester.monitor_timeout, 2)
        self.assertAlmostEqual(tester.threshold, 3.14, places=2)
        self.assertAlmostEqual(tester.bias_samples._target_elapsed_time, 30.0)
        self.assertAlmostEqual(tester.max_delta_t, 40.0)

    def test_from_config(self):
        config = {
            'receiver_id': TEST_RX_STR,
            "threshold": 0.5,
            "monitor_timeout": 60,
            'min_delta_t': 2.1,
            'max_delta_t': 3.2
        }

        tester = monitor.from_config('CCDMonitor', config)

        self.assertEqual(tester.receiver_id, TEST_RX_STR)
        self.assertAlmostEqual(tester.threshold, 0.5)
        self.assertEqual(tester.monitor_timeout, 60)
        self.assertAlmostEqual(tester.min_delta_t, 2.1)
        self.assertAlmostEqual(tester.max_delta_t, 3.2)

    def testMinDeltaTSetter(self):
        tester = ccd_monitor.CCDMonitor(receiver_id=TEST_RX_STR, threshold=0.0015, min_delta_t=5, max_delta_t=10)
        tester.min_delta_t = 20
        self.assertEqual(tester.bias_samples._target_elapsed_time, 20)

        tester.min_delta_t = 15
        self.assertEqual(tester.bias_samples._target_elapsed_time, 15)

        tester.min_delta_t = 5.5
        self.assertEqual(tester.bias_samples._target_elapsed_time, 5.5)

        # Invalid numbers leave min_delta_t unchanged
        with self.assertRaises(ValueError):
            tester.min_delta_t = -2

    def testMaxDeltaTSetter(self):
        tester = ccd_monitor.CCDMonitor(receiver_id=TEST_RX_STR, threshold=0.0015, min_delta_t=5, max_delta_t=10)
        tester.max_delta_t = 20
        self.assertEqual(tester.max_delta_t, 20)

        tester.max_delta_t = 25
        self.assertEqual(tester.max_delta_t, 25)

        tester.max_delta_t = 5.5
        self.assertEqual(tester.max_delta_t, 5.5)

        with self.assertRaises(ValueError):
            tester.max_delta_t = -2

    def testThresholdSetter(self):
        tester = ccd_monitor.CCDMonitor(receiver_id=TEST_RX_STR, threshold=0.0015, min_delta_t=5, max_delta_t=10)
        tester.threshold = 0.001
        self.assertEqual(tester.threshold, 0.001)

        tester.threshold = 0.0015
        self.assertEqual(tester.threshold, 0.0015)

        with self.assertRaises(ValueError):
            tester.threshold = -2

    def test_calc_metric1(self):
        tester = ccd_monitor.CCDMonitor(receiver_id=0, threshold=43.6, min_delta_t=5, max_delta_t=6)
        self.assertAlmostEqual(tester._threshold, 43.6)

        message = copy.deepcopy(BASIC_MESSAGE)

        times = [1., 2., 3., 4., 5., 6.]

        message['clock_bias'] = 1.
        message['clock_rate'] = 1.

        # Not enough time elapsed, so metric should be None
        for time in times:
            message['rxTime'] = time
            res = tester._calculate_metric(message)
            self.assertIsNone(res)
            # Since bias is one, the sum tracks the time after first time
            # skipped
            self.assertAlmostEqual(tester.bias_samples.sum, time - 1)

            # This isn't defined until at least 2 samples have been processed
            if time > 1.2:
                # After processing one message, this should stay set to 1.
                self.assertAlmostEqual(tester._last_rate, 1.)
                self.assertAlmostEqual(tester.rate_integral.sum, time - 1)

            else:
                self.assertEqual(len(tester.rate_integral), 0)

        # Metric should be zero--now there is enough time accumulated
        message['rxTime'] = 7.
        self.assertAlmostEqual(tester._calculate_metric(message), 0.0)

    def test_calc_metric2(self):
        tester = ccd_monitor.CCDMonitor(receiver_id=0, threshold=43.6, min_delta_t=5, max_delta_t=6)

        message = copy.deepcopy(BASIC_MESSAGE)

        times = [0., 1., 2., 3., 4., 5., 6]
        biases = [2., 2., 2., 2., 1., 3., 1.]
        # Bias sums will omit the first element
        bias_sums = [0., 2., 4., 6., 7., 10., 11.]
        rates = [1., 1., 1., 1., 1., 1., 1.]

        for (time, bias, rate, bias_sum) in zip(times, biases, rates, bias_sums):
            message['rxTime'] = time
            message['clock_bias'] = bias
            message['clock_rate'] = rate

            if time < 6.:
                self.assertIsNone(tester._calculate_metric(message))
            else:
                self.assertAlmostEqual(tester._calculate_metric(message), 5.)

            if time > 0.1:
                self.assertAlmostEqual(tester.rate_integral.sum, time)  # Since rate is 1, the integral tracks the time

            self.assertAlmostEqual(tester._last_rate, 1.)  # After processing first message, this should stay 1.

            self.assertAlmostEqual(tester.bias_samples.sum, bias_sum)

    def test_calc_metric3(self):
        tester = ccd_monitor.CCDMonitor(receiver_id=0, threshold=43.6, min_delta_t=5, max_delta_t=6)

        message = copy.deepcopy(BASIC_MESSAGE)

        times = [0., 1., 2., 3., 4., 5., 6.]
        biases = [1., 1., 1., 1., 1., 1., 1.]
        rates = [1., 2., 3., 4., 5., 6., 7.]
        integrals = [None, 1.5, 4., 7.5, 12., 17.5, 24]

        for (time, bias, rate, integral) in zip(times, biases, rates, integrals):
            message['rxTime'] = time
            message['clock_bias'] = bias
            message['clock_rate'] = rate

            if time < 6.:
                self.assertIsNone(tester._calculate_metric(message))
            else:
                self.assertAlmostEqual(tester._calculate_metric(message), 18)

            if time > 0.1:
                self.assertAlmostEqual(tester.rate_integral.sum, integral)
                self.assertAlmostEqual(tester._last_rate, rate)  # After processing first message, this should stay 1.

            self.assertAlmostEqual(tester.bias_samples.sum, time)

    def test_const_rate_nonspoofed(self):
        tester = ccd_monitor.CCDMonitor(receiver_id=TEST_RX_STR, threshold=43.6, min_delta_t=20, max_delta_t=20.1)

        message = copy.deepcopy(BASIC_MESSAGE)
        message['clock_bias'] = 5
        message['clock_rate'] = 5

        for time in range(50):
            message['rxTime'] = time

            if time < 21:
                self.assertIsNone(tester.update(message))
                self.assertFalse(tester._status['alarm'])
            else:
                self.assertFalse(tester.update(message))
                self.assertFalse(tester._status['alarm'])
                self.assertAlmostEqual(tester.metric, 0.0)

        self.assertFalse(tester._status['alarm'])

    def testSinusoidalCdrDotNonspoofed(self):
        tester = ccd_monitor.CCDMonitor(receiver_id=TEST_RX_STR, threshold=43.6, min_delta_t=20, max_delta_t=20.1)

        message = copy.deepcopy(BASIC_MESSAGE)

        for (time, bias, rate) in self.get_test_data('ccd_tv_sinusoidal_cdr_dot_nonspoofed'):
            message['rxTime'] = time
            message['clock_bias'] = bias
            message['clock_rate'] = rate

            tester.update(message)

        self.assertFalse(tester._status['alarm'])

    # TODO: This test or its data may need updating now that the FIR filters are gone
    # def testCdrPulloffSpoofed(self):
    #     tester = ccd_monitor.CCDMonitor(receiver_id=TEST_RX_STR, threshold=43.6, min_delta_t=20, max_delta_t=20.1)
    #
    #     message = copy.deepcopy(BASIC_MESSAGE)
    #
    #     for (time, bias, rate) in self.get_test_data('ccd_tv_sinusoidal_cdr_dot_nonspoofed'):
    #         message['rxTime'] = time
    #         message['clock_bias'] = bias
    #         message['clock_rate'] = rate
    #
    #         tester.update(message)
    #
    #     self.assertTrue(tester._status['alarm'])

    def testNoClockData(self):
        tester = ccd_monitor.CCDMonitor(receiver_id=TEST_RX_STR, threshold=43.6, min_delta_t=5, max_delta_t=6)

        self.assertIsNone(tester.update(BASIC_MESSAGE))
        self.assertEqual(len(tester.bias_samples), 0)

    def test_reset(self):
        """
        @brief Ensures the filter timeout resets the FIR filter and nothing
               else.
        """

        tester = ccd_monitor.CCDMonitor(receiver_id=TEST_RX_STR, threshold=43.6, min_delta_t=5, max_delta_t=6)
        times = [0., 1., 2., 3., 4., 5.]
        biases = [1., 1., 1., 1., 1., 1.]
        rates = [1., 2., 3., 4., 5., 6.]

        message = copy.deepcopy(BASIC_MESSAGE)

        for (time, bias, rate) in zip(times, biases, rates):
            message['rxTime'] = time
            message['clock_bias'] = bias
            message['clock_rate'] = rate

            tester.update(message)

        self.assertNotEqual(len(tester.bias_samples), 0)
        self.assertNotEqual(len(tester.rate_integral), 0)
        self.assertNotEqual(tester.bias_samples.sum, 0)
        self.assertNotEqual(tester.rate_integral.sum, 0)
        self.assertIsNotNone(tester._last_rate)

        tester.reset()

        self.assertEqual(len(tester.bias_samples), 0)
        self.assertEqual(len(tester.rate_integral), 0)
        self.assertEqual(tester.bias_samples.sum, 0)
        self.assertEqual(tester.rate_integral.sum, 0)
        self.assertIsNone(tester._last_rate)


class TestDualAntennaDistanceMonitor(unittest.TestCase):
    def test_create(self):
        tester = dadm.DualAntennaDistanceMonitor(receiver_id_1='Test Rx 1', receiver_id_2='Test Rx 2')

        self.assertEqual(tester.rx1.receiver_id, 'Test Rx 1')
        self.assertEqual(tester.rx2.receiver_id, 'Test Rx 2')
        self.assertEqual(tester.monitor_timeout, None)
        self.assertEqual(tester._threshold, 2)
        self.assertEqual(tester._min_samples, 10)
        self.assertEqual(tester.rx1.samples._target_elapsed_time, 15)
        self.assertEqual(tester.rx2.samples._target_elapsed_time, 15)

    def test_from_config(self):
        config = {
            'receiver_id_1': 'Test Rx 1',
            'receiver_id_2': 'Test Rx 2',
            "threshold": 0.5,
            "monitor_timeout": 60,
            'minimum_samples': 12,
            'time_range': 12.1
        }

        tester = monitor.from_config('DualAntennaDistanceMonitor', config)

        self.assertEqual(tester.rx1.receiver_id, 'Test Rx 1')
        self.assertEqual(tester.rx2.receiver_id, 'Test Rx 2')
        self.assertEqual(tester.monitor_timeout, 60)
        self.assertAlmostEqual(tester._threshold, 0.5)
        self.assertEqual(tester._min_samples, 12)
        self.assertAlmostEqual(tester.rx1.samples._target_elapsed_time, 12.1)
        self.assertAlmostEqual(tester.rx2.samples._target_elapsed_time, 12.1)

    def testTimeRangeSetter(self):
        tester = dadm.DualAntennaDistanceMonitor(receiver_id_1='Test Rx 1', receiver_id_2='Test Rx 2')

        tester.time_range = 25
        self.assertEqual(tester.time_range, 25)

        with self.assertRaises(ValueError):
            tester.time_range = -5

    def test_compare_metric(self):
        tester = dadm.DualAntennaDistanceMonitor(receiver_id_1='Test Rx 1', receiver_id_2='Test Rx 2', threshold=3.14)

        self.assertFalse(tester._compare_metric(27))
        self.assertTrue(tester._compare_metric(0.34))

    def testAddEventToBuffers(self):
        tester = dadm.DualAntennaDistanceMonitor(receiver_id_1='Test Rx 1', receiver_id_2='Test Rx 2')

        message = copy.deepcopy(BASIC_MESSAGE)
        message['ecef_position'] = [0, 0, 0]
        message['receiver_id'] = 'Test Rx 1'

        self.assertIsNone(tester.update(message))   # First message won't have enough information to determine a metric

        self.assertEqual(len(tester.rx1.samples), 1)
        self.assertEqual(len(tester.rx2.samples), 0)

        message['receiver_id'] = 'Test Rx 2'
        message['rxTime'] += 1

        self.assertIsNone(tester.update(message))  # Second message won't have enough information to determine a metric
        self.assertEqual(len(tester.rx1.samples), 1)
        self.assertEqual(len(tester.rx2.samples), 1)

        # Update in reverse order
        message['receiver_id'] = 'Test Rx 2'
        message['rxTime'] += 1

        self.assertIsNone(tester.update(message))  # Still not enough info (both receivers have samples but not enough)
        self.assertEqual(len(tester.rx1.samples), 1)
        self.assertEqual(len(tester.rx2.samples), 2)

        message['receiver_id'] = 'Test Rx 1'
        message['rxTime'] += 1

        self.assertIsNone(tester.update(message))  # Still not enough info (both receivers have samples but not enough)
        self.assertEqual(len(tester.rx1.samples), 2)
        self.assertEqual(len(tester.rx2.samples), 2)

    def testComputeMetric(self):
        tester = dadm.DualAntennaDistanceMonitor(receiver_id_1='Test Rx 1', receiver_id_2='Test Rx 2',
                                                 threshold=2, time_range=25.0)

        # Initialize the monitor buffers with known values.
        message1 = copy.deepcopy(BASIC_MESSAGE)
        message2 = copy.deepcopy(BASIC_MESSAGE)

        message1['receiver_id'] = 'Test Rx 1'
        message2['receiver_id'] = 'Test Rx 2'

        for ix in range(20):
            message1['ecef_position'] = [ix, ix**2, ix**3]
            tester.update(message1)
            message1['rxTime'] += 1

            message2['ecef_position'] = [2*ix, 2*ix, 3*ix**2]
            tester.update(message2)
            message2['rxTime'] += 1

        self.assertAlmostEqual(tester.metric, 1438.3326284277916)

        # Send a few more messages to only one monitor and make sure the metric doesn't change since the other hasn't
        #  been updated
        for ix in range(20, 25):
            message1['ecef_position'] = [ix, ix ** 2, ix ** 3]
            tester.update(message1)
            message1['rxTime'] += 1

        self.assertAlmostEqual(tester.metric, 1438.3326284277916)

    def test_reset(self):
        tester = dadm.DualAntennaDistanceMonitor(receiver_id_1='Test Rx 1', receiver_id_2='Test Rx 2',
                                                 threshold=2, time_range=25.0)

        # Initialize the monitor buffers with known values.
        message1 = copy.deepcopy(BASIC_MESSAGE)
        message2 = copy.deepcopy(BASIC_MESSAGE)

        message1['receiver_id'] = 'Test Rx 1'
        message2['receiver_id'] = 'Test Rx 2'

        for ix in range(25):
            message1['ecef_position'] = [ix, ix ** 2, ix ** 3]
            tester.update(message1)
            message1['rxTime'] += 1

            if ix < 20:
                message2['ecef_position'] = [2 * ix, 2 * ix, 3 * ix ** 2]
                tester.update(message2)
                message2['rxTime'] += 1

        self.assertAlmostEqual(tester.metric, 1438.3326284277916)
        self.assertEqual(len(tester.rx1.samples), 25)
        self.assertEqual(len(tester.rx2.samples), 20)

        tester.reset()

        self.assertIsNone(tester.metric)
        self.assertEqual(len(tester.rx1.samples), 0)
        self.assertEqual(len(tester.rx2.samples), 0)
        self.assertIsNone(tester._last_update)


class TestCnoThresholdJammingMonitor(unittest.TestCase):
    def test_create(self):
        tester = cn0_threshold_monitor.CnoThresholdJammingMonitor(receiver_id=TEST_RX_STR, threshold=20,
                                                                  time_window=5)

        self.assertEqual(tester.receiver_id, TEST_RX_STR)
        self.assertEqual(tester.threshold, 20)
        self.assertEqual(tester.time_window, 5)

    def testTimeWindowSetter(self):
        tester = cn0_threshold_monitor.CnoThresholdJammingMonitor(receiver_id=TEST_RX_STR, threshold=20,
                                                                  time_window=5)

        self.assertEqual(tester.time_window, 5)

        tester.time_window = 15

        self.assertEqual(tester.time_window, 15)

        with self.assertRaises(ValueError):
            tester.time_window = -5

    def test_create_from_config(self):
        # Use the thin wrapper, so this test configuration will only work in this test environment
        config = {
            'receiver_id': TEST_RX_STR,
            'threshold': 20,
            'time_window': 5
        }

        tester = monitor.from_config(monitor_name='CnoThresholdJammingMonitor', configuration=config)

        self.assertIsInstance(tester, cn0_threshold_monitor.CnoThresholdJammingMonitor)
        self.assertEqual(tester.receiver_id, TEST_RX_STR)
        self.assertIsNone(tester.monitor_timeout)
        self.assertEqual(tester._threshold, 20)
        self.assertEqual(tester.time_window, 5)

    def testNoCnoData(self):
        message = BASIC_MESSAGE
        tester = cn0_threshold_monitor.CnoThresholdJammingMonitor(receiver_id=TEST_RX_STR, threshold=20,
                                                                  time_window=5)

        self.assertIsNone(tester.update(message))

    def test_update(self):
        tester = cn0_threshold_monitor.CnoThresholdJammingMonitor(receiver_id=TEST_RX_STR, threshold=20,
                                                                  time_window=5)

        message = copy.deepcopy(BASIC_MESSAGE)
        message['svs'] = [
            {'gnssId': 0, 'svid': 1, 'cno': 15, 'qualityInd': 5},
            {'gnssId': 0, 'svid': 2, 'cno': 15, 'qualityInd': 5}
        ]

        message['rxTime'] = 1

        self.assertTrue(tester.update(message))
        self.assertEqual(tester.metric, 15)
        self.assertEqual(len(tester._cnos), 2)
        self.assertTrue('0.1' in tester._cnos)
        self.assertTrue('0.2' in tester._cnos)

        self.assertEqual(tester._cnos['0.1'], (1, 15))
        self.assertEqual(tester._cnos['0.2'], (1, 15))

        message['svs'][0]['cno'] = 25
        message['rxTime'] += 1

        self.assertFalse(tester.update(message))
        self.assertEqual(tester.metric, 25)
        self.assertEqual(tester._cnos['0.1'], (2, 25))
        self.assertEqual(tester._cnos['0.2'], (2, 15))

    def testTimeWindow(self):
        tester = cn0_threshold_monitor.CnoThresholdJammingMonitor(receiver_id=TEST_RX_STR, threshold=20,
                                                                  time_window=5)

        message = copy.deepcopy(BASIC_MESSAGE)
        message['svs'] = [
            {'gnssId': 0, 'svid': 1, 'cno': 25, 'qualityInd': 5},
            {'gnssId': 0, 'svid': 2, 'cno': 15, 'qualityInd': 5}
        ]

        message['rxTime'] = 1

        self.assertFalse(tester.update(message))
        self.assertEqual(tester.metric, 25)
        self.assertEqual(len(tester._cnos), 2)
        self.assertTrue('0.1' in tester._cnos)
        self.assertTrue('0.2' in tester._cnos)

        message['svs'] = [
            {'gnssId': 0, 'svid': 2, 'cno': 2, 'qualityInd': 5}
        ]

        message['rxTime'] = 82

        self.assertTrue(tester.update(message))
        self.assertEqual(tester.metric, 2)
        self.assertEqual(len(tester._cnos), 2)
        self.assertTrue('0.1' in tester._cnos)
        self.assertTrue('0.2' in tester._cnos)
        self.assertIsNone(tester._cnos['0.1'])
        self.assertEqual(tester._cnos['0.2'], (82, 2))


class TestCnoSpoofingMonitor(unittest.TestCase):
    def test_create(self):
        tester = cn0_spoofing_monitor.CnoSpoofingMonitor(receiver_id=TEST_RX_STR, channel_id='0.1', threshold=40)

        self.assertEqual(tester.receiver_id, TEST_RX_STR)
        self.assertEqual(tester.threshold, 40)
        self.assertEqual(tester._gnss_id, 0)
        self.assertEqual(tester._sv_id, 1)

    def test_create_from_config(self):
        # Use the thin wrapper, so this test configuration will only work in this test environment
        config = {
            'receiver_id': TEST_RX_STR,
            'channel_id': '1.2',
            'threshold': 51
        }

        tester = monitor.from_config(monitor_name='CnoSpoofingMonitor', configuration=config)

        self.assertIsInstance(tester, cn0_spoofing_monitor.CnoSpoofingMonitor)
        self.assertEqual(tester.receiver_id, TEST_RX_STR)
        self.assertEqual(tester._threshold, 51)
        self.assertEqual(tester._gnss_id, 1)
        self.assertEqual(tester._sv_id, 2)

    def testIncorrectChannelId(self):
        tester = cn0_spoofing_monitor.CnoSpoofingMonitor(receiver_id=TEST_RX_STR, channel_id='0.1', threshold=40)

        message = copy.deepcopy(BASIC_MESSAGE)
        message['svs'] = [
            {'gnssId': 1, 'svid': 1, 'cno': 12, 'qualityInd': 5}
        ]

        self.assertIsNone(tester.update(message))

    def testNoCnoData(self):
        tester = cn0_spoofing_monitor.CnoSpoofingMonitor(receiver_id=TEST_RX_STR, channel_id='0.1', threshold=40)

        message = copy.deepcopy(BASIC_MESSAGE)

        self.assertIsNone(tester.update(message))

        message['svs'] = [
            {'gnssId': 1, 'svid': 1, 'qualityInd': 5}
        ]

        self.assertIsNone(tester.update(message))

    def test_update(self):
        tester = cn0_spoofing_monitor.CnoSpoofingMonitor(receiver_id=TEST_RX_STR, channel_id='0.1', threshold=40)

        message = copy.deepcopy(BASIC_MESSAGE)
        message['rxTime'] = 1
        message['svs'] = [
            {'gnssId': 0, 'svid': 1, 'cno': 39, 'qualityInd': 5}
        ]

        self.assertFalse(tester.update(message))
        self.assertEqual(tester.metric, 39)

        message['rxTime'] = 2
        message['svs'][0]['cno'] = 41

        self.assertTrue(tester.update(message))
        self.assertEqual(tester.metric, 41)


class TestCnoDropJammingMonitor(unittest.TestCase):
    def test_create(self):
        tester = cn0_drop_monitor.CnoDropJammingMonitor(receiver_id=TEST_RX_STR, threshold=5, time_window=5)

        self.assertEqual(tester.receiver_id, TEST_RX_STR)
        self.assertEqual(tester.threshold, 5)
        self.assertEqual(tester.time_window, 5)

    def testTimeWindowSetter(self):
        tester = tester = cn0_drop_monitor.CnoDropJammingMonitor(receiver_id=TEST_RX_STR, threshold=5, time_window=5)

        self.assertEqual(tester.time_window, 5)

        tester.time_window = 15

        self.assertEqual(tester.time_window, 15)

        with self.assertRaises(ValueError):
            tester.time_window = -5

    def test_create_from_config(self):
        # Use the thin wrapper, so this test configuration will only work in this test environment
        config = {
            'receiver_id': TEST_RX_STR,
            'threshold': 20,
            'time_window': 5
        }

        tester = monitor.from_config(monitor_name='CnoDropJammingMonitor', configuration=config)

        self.assertIsInstance(tester, cn0_drop_monitor.CnoDropJammingMonitor)
        self.assertEqual(tester.receiver_id, TEST_RX_STR)
        self.assertIsNone(tester.monitor_timeout)
        self.assertEqual(tester._threshold, 20)
        self.assertEqual(tester.time_window, 5)

    def testComputeCnoDrops(self):
        tester = cn0_drop_monitor.CnoDropJammingMonitor(receiver_id=TEST_RX_STR, threshold=5, time_window=5)

        message = copy.deepcopy(BASIC_MESSAGE)
        message['rxTime'] = 1
        message['svs'] = [
            {'gnssId': 0, 'svid': 1, 'cno': 12, 'qualityInd': 5},
            {'gnssId': 0, 'svid': 0, 'cno': 13, 'qualityInd': 5},
            {'gnssId': 1, 'svid': 1, 'cno': 9, 'qualityInd': 5}
        ]

        # Not enough data for update to return True or False
        self.assertIsNone(tester.update(message))
        self.assertIsNone(tester.metric)
        self.assertEqual(len(tester._cnos), 3)
        self.assertEqual(len(tester._drops), 3)  # Not enough data yet
        for channel in tester._cnos:
            self.assertEqual(len(tester._cnos[channel]), 1)

        message['rxTime'] += 1
        message['svs'][0]['cno'] = 10   # A drop but not enough to trigger

        # No alarm
        self.assertFalse(tester.update(message))
        self.assertEqual(len(tester._drops), 3)
        self.assertFalse(tester.metric)
        self.assertDictEqual(tester._drops, {'0.1': 2, '0.0': 0, '1.1': 0})
        for channel in tester._cnos:
            self.assertEqual(len(tester._cnos[channel]), 2)

        message['rxTime'] += 1
        message['svs'][1]['cno'] = 7  # A drop beyond the threshold but the others are not so no alarm

        # No alarm but one beyond threshold
        self.assertFalse(tester.update(message))
        self.assertFalse(tester.metric)
        self.assertEqual(len(tester._drops), 3)
        self.assertDictEqual(tester._drops, {'0.1': 2, '0.0': 6, '1.1': 0})
        for channel in tester._cnos:
            self.assertEqual(len(tester._cnos[channel]), 3)

        # Drop all
        message['rxTime'] += 1
        message['svs'][0]['cno'] = 1
        message['svs'][1]['cno'] = 1
        message['svs'][2]['cno'] = 1

        # Now there's an alarm
        self.assertTrue(tester.update(message))
        self.assertTrue(tester.metric)
        self.assertEqual(len(tester._drops), 3)
        self.assertDictEqual(tester._drops, {'0.1': 11, '0.0': 12, '1.1': 8})
        for channel in tester._cnos:
            self.assertEqual(len(tester._cnos[channel]), 4)

        # Advance enough time that all but the last samples expire
        message['rxTime'] += 5.1
        message['svs'][0]['cno'] = 1
        message['svs'][1]['cno'] = 1
        message['svs'][2]['cno'] = 1

        self.assertFalse(tester.update(message))
        self.assertFalse(tester.metric)
        self.assertEqual(len(tester._drops), 3)
        self.assertDictEqual(tester._drops, {'0.1': 0, '0.0': 0, '1.1': 0})
        for channel in tester._cnos:
            self.assertEqual(len(tester._cnos[channel]), 2)


# class TestAgcMonitor(unittest.TestCase):
#     def setUp(self):
#         self.longMessage = True
#         tester = AgcMonitor(receiver_id=1, threshold=200)
#
#     def testInvalidEvent(self):
#         event = PowerEvent(agc=250)
#         metadata1 = EventMetadata(receiver_id=1, event_time=1)
#         metadata2 = EventMetadata(receiver_id=1, event_time=2)
#         event.validity = True
#         self.assertTrue(tester.processEvent(event, metadata1).validity)
#
#         event.validity = False
#         self.assertFalse(tester.processEvent(event, metadata2).validity)
#
#     def testStringReceiverId(self):
#         monitor = AgcMonitor(receiver_id='a', threshold=200)
#
#         event = PowerEvent(agc=45)
#
#         metadata1 = EventMetadata(receiver_id='a', event_time=1)
#         metadata2 = EventMetadata(receiver_id='b', event_time=1)
#
#         out = monitor.processEvent(event, metadata1)
#         self.assertTrue(out.validity)
#
#         out = monitor.processEvent(event, metadata2)
#         self.assertFalse(out.validity)
#
#     def testJSONInitialize(self):
#
#         json_string = """{
#         "AgcMonitor":
#         {
#             "threshold": 200
#         }
#         }"""
#         monitor = Monitor.createMonitor(json_string, receiver_id=1)
#
#         self.assertEqual(monitor.receiver_id, 1)
#         self.assertEqual(monitor.threshold, 200)
#
#     def testIncorrectReceiverId(self):
#         event = PowerEvent(agc=50)
#         metadatas = generateEventMetadata([1, 1], [1, 2])
#
#         self.assertTrue(
#             tester.processEvent(event, metadatas[0]).validity)
#         self.assertFalse(
#             tester.processEvent(event, metadatas[1]).validity)
#
#     def testNoAgcData(self):
#         event1 = PowerEvent(agc=5)
#         event2 = PowerEvent()
#         metadata = EventMetadata(receiver_id=1, event_time=1)
#         self.assertTrue(
#             tester.processEvent(event1, metadata).validity)
#         self.assertFalse(
#             tester.processEvent(event2, metadata).validity)
#
#     def testOldEvents(self):
#         event = PowerEvent(agc=50)
#         metadatas = generateEventMetadata([2, 1], [1, 1])
#
#         self.assertTrue(
#             tester.processEvent(event, metadatas[0]).validity)
#         self.assertFalse(
#             tester.processEvent(event, metadatas[1]).validity)
#
#     def testThreshold(self):
#         events = generatePowerEvents([199, 201])
#         metadata = EventMetadata(receiver_id=1, event_time=1)
#
#         out = tester.processEvent(events[0], metadata)
#         self.assertTrue(out.validity)
#         self.assertEqual(out.integrity_status, StatusType.VALID)
#         self.assertEqual(out.metric, 199)
#
#         out = tester.processEvent(events[1], metadata)
#         self.assertTrue(out.validity)
#         self.assertNotEqual(out.integrity_status, StatusType.VALID)
#         self.assertEqual(out.metric, 201)


if __name__ == '__main__':
    unittest.main()
