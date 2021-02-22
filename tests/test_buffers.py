# -*- coding: utf-8 -*-
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
"""
Unit tests for buffers.py.
"""

import unittest
from epsilon.buffers import (FIFORunningSum,
                             FIRFilter,
                             TimedBuffer,
                             TimedRunningSum)


class TestFIFORunningSum(unittest.TestCase):
    def test_create(self):
        tester = FIFORunningSum(3)

        self.assertEqual(tester._buffer.maxlen, 3)
        self.assertEqual(tester._sum, 0)

    def test_sum_property(self):
        fifo = FIFORunningSum(3)

        self.assertEqual(fifo.sum, 0)

        with self.assertRaises(AttributeError):
            fifo.sum = 5

    def test_max_len(self):
        tester = FIFORunningSum(8)
        self.assertEqual(tester.maxlen, 8)

    def test_append(self):
        fifo = FIFORunningSum(3)

        fifo.append(1)
        self.assertEqual(fifo.sum, 1)

        fifo.append(2)
        self.assertEqual(fifo.sum, 3)

        fifo.append(3)
        self.assertEqual(fifo.sum, 6)

        fifo.append(4)
        self.assertEqual(fifo.sum, 9)

        fifo.append(5)
        self.assertEqual(fifo.sum, 12)

        fifo.append(6)
        self.assertEqual(fifo.sum, 15)

    def test_len(self):
        fifo = FIFORunningSum(8)

        self.assertEqual(len(fifo), 0)

        fifo.append(1.0)
        self.assertEqual(len(fifo), 1)

        self.assertEqual(fifo._buffer.maxlen, 8)

    def test_fifo_reset(self):
        fifo = FIFORunningSum(3)

        fifo.append(1)
        fifo.append(2)
        fifo.append(3)

        fifo.reset()

        self.assertEqual(fifo.sum, 0)
        self.assertEqual(fifo._buffer.maxlen, 3)
        self.assertEqual(len(fifo), 0)

    def test_logical_values(self):
        fifo = FIFORunningSum(3)

        fifo.append(True)
        fifo.append(False)
        fifo.append(True)

        self.assertEqual(fifo.sum, 2)


class TestFIRFilter(unittest.TestCase):
    def test_create(self):
        tester = FIRFilter([1, 2, 3])

        self.assertEqual(tester._past_samples.maxlen, 3)
        self.assertEqual(len(tester._past_samples), 0)

        self.assertEqual(tester._filter_length, 3)
        self.assertListEqual(tester._coefficients, [1, 2, 3])
        self.assertFalse(tester.is_initialized)

    def test_append(self):
        tester = FIRFilter([1, 2, 3])

        tester.appendleft(4)

        self.assertFalse(tester.is_initialized)
        self.assertEqual(len(tester._past_samples), 1)
        self.assertEqual(tester._past_samples.maxlen, 3)
        self.assertIsNone(tester._response)     # Not enough samples yet to get a response

        tester.appendleft(5)

        self.assertFalse(tester.is_initialized)
        self.assertEqual(len(tester._past_samples), 2)
        self.assertEqual(tester._past_samples.maxlen, 3)
        self.assertIsNone(tester._response)  # Not enough samples yet to get a response

        tester.appendleft(6)

        self.assertTrue(tester.is_initialized)  # Now there are enough samples
        self.assertEqual(len(tester._past_samples), 3)
        self.assertEqual(tester._past_samples.maxlen, 3)
        self.assertEqual(tester._response, 6 * 1 + 5 * 2 + 4 * 3)  # Now the filter is initialized

        # First element should get evicted and the rest rotated
        tester.appendleft(7)

        self.assertTrue(tester.is_initialized)
        self.assertEqual(len(tester._past_samples), 3)
        self.assertEqual(tester._past_samples.maxlen, 3)
        self.assertEqual(tester._response, 7 * 1 + 6 * 2 + 5 * 3)

        # Second sample (now first element) should get evicted and the rest rotated
        tester.appendleft(8)

        self.assertTrue(tester.is_initialized)
        self.assertEqual(len(tester._past_samples), 3)
        self.assertEqual(tester._past_samples.maxlen, 3)
        self.assertEqual(tester._response, 8 * 1 + 7 * 2 + 6 * 3)

        # Third element should get evicted and the rest rotated; all original elements should be gone
        tester.appendleft(9)

        self.assertTrue(tester.is_initialized)
        self.assertEqual(len(tester._past_samples), 3)
        self.assertEqual(tester._past_samples.maxlen, 3)
        self.assertEqual(tester._response, 9 * 1 + 8 * 2 + 7 * 3)

    def test_impulse_response(self):
        coefficients = [3., 2., 1.]
        filt = FIRFilter(coefficients)

        data = [0, 0, 0, 1, 0, 0, 0]
        solution = [0, 0, 0, 3, 2, 1, 0]

        for d, s in zip(data, solution):
            filt.appendleft(d)
            if filt.is_initialized:
                self.assertAlmostEqual(filt.response, s)

    def test_step_response(self):
        coefficients = [1., 4., 2.]
        filt = FIRFilter(coefficients)

        data = [0, 0, 0, 1, 1, 1, 1]
        solution = [0, 0, 0, 1, 5, 7, 7]

        for d, s in zip(data, solution):
            filt.appendleft(d)
            if filt.is_initialized:
                self.assertAlmostEqual(filt.response, s)

    def test_length(self):
        tester = FIRFilter([1, 2, 3])

        self.assertEqual(len(tester), 0)

        tester.appendleft(1)
        self.assertEqual(len(tester), 1)

    def test_reset(self):
        coefficients = [1., 4., 2.]
        filt = FIRFilter(coefficients)

        filt.appendleft(1)
        filt.appendleft(2)
        filt.appendleft(3)

        self.assertEqual(len(filt), 3)
        self.assertAlmostEqual(filt.response, 13)
        self.assertTrue(filt.is_initialized)

        filt.reset()

        self.assertFalse(filt.is_initialized)
        self.assertEqual(len(filt), 0)
        self.assertIsNone(filt.response)


class TestTimedBuffer(unittest.TestCase):
    def test_create(self):
        tester = TimedBuffer(target_elapsed_time=3)

        self.assertEqual(tester._target_elapsed_time, 3)

    def test_remove_old_samples(self):
        tester = TimedBuffer(10)

        tester._time_buffer.extend([1, 2, 13])
        tester._sample_buffer.extend([4, 5, 6])

        self.assertEqual(tester._time_buffer[0], 1)
        self.assertEqual(tester._time_buffer[1], 2)
        self.assertEqual(tester._time_buffer[2], 13)

        self.assertEqual(tester._sample_buffer[0], 4)
        self.assertEqual(tester._sample_buffer[1], 5)
        self.assertEqual(tester._sample_buffer[2], 6)

        tester.remove_old_samples()

        self.assertEqual(len(tester), 2)
        self.assertEqual(tester._time_buffer[0], 2)
        self.assertEqual(tester._sample_buffer[0], 5)

        self.assertEqual(tester._time_buffer[1], 13)
        self.assertEqual(tester._sample_buffer[1], 6)

    def test_append(self):
        tester = TimedBuffer(10)
        tester.append(1, 1)

        self.assertEqual(len(tester), 1)
        self.assertEqual(tester._time_buffer[0], 1)
        self.assertEqual(tester._sample_buffer[0], 1)

        tester.append(2, 2)

        self.assertEqual(len(tester), 2)
        self.assertEqual(tester._time_buffer[0], 1)
        self.assertEqual(tester._sample_buffer[0], 1)
        self.assertEqual(tester._time_buffer[1], 2)
        self.assertEqual(tester._sample_buffer[1], 2)

        tester.append(15, 4)

        # Will now have 2 elements because one older sample is kept
        self.assertEqual(len(tester), 2)
        self.assertEqual(tester._time_buffer[0], 2)
        self.assertEqual(tester._sample_buffer[0], 2)
        self.assertEqual(tester._time_buffer[1], 15)
        self.assertEqual(tester._sample_buffer[1], 4)

    def test_popleft(self):
        tester = TimedBuffer(10)
        tester.append(1, 1)

        self.assertEqual(len(tester), 1)
        self.assertEqual(tester._time_buffer[0], 1)
        self.assertEqual(tester._sample_buffer[0], 1)

        tester.popleft()

        self.assertEqual(len(tester), 0)

    def test_get_oldest(self):
        tester = TimedBuffer(10)

        self.assertIsNone(tester.oldest_sample)

        tester.append(1, 2)
        self.assertEqual(tester.oldest_sample, 2)

        tester.append(10, 4)
        self.assertEqual(tester.oldest_sample, 2)

        tester.append(11.1, 15)  # No evictions yet because one older sample is kept
        self.assertEqual(tester.oldest_sample, 2)

        tester.append(11.2, 32)  # Still no eviction: 10 to 11.2 is within the window of 10
        self.assertEqual(tester.oldest_sample, 2)

        tester.append(21, 32)  # Now there should be an eviction but the (10, 4) will still be there
        self.assertEqual(tester.oldest_sample, 4)

    def test_get_newest(self):
        tester = TimedBuffer(10)

        self.assertIsNone(tester.newest_sample)

        tester.append(1, 2)
        self.assertEqual(tester.newest_sample, 2)

        tester.append(10, 4)
        self.assertEqual(tester.newest_sample, 4)

        tester.append(11.1, 15)
        self.assertEqual(tester.newest_sample, 15)

    def test_get_last_time(self):
        tester = TimedBuffer(10)

        self.assertEqual(tester.newest_time, 0)

        tester.append(1, 2)
        self.assertEqual(tester.newest_time, 1)

        tester.append(10, 8)
        self.assertEqual(tester.newest_time, 10)

        tester.append(11.1, 15)
        self.assertEqual(tester.newest_time, 11.1)

    def test_get_oldest_time(self):
        tester = TimedBuffer(10)

        self.assertEqual(tester.oldest_time, 0)

        tester.append(1, 2)
        self.assertEqual(tester.oldest_time, 1)

        tester.append(10, 8)
        self.assertEqual(tester.oldest_time, 1)

        tester.append(11.1, 15)     # No eviction
        self.assertEqual(tester.oldest_time, 1)

        tester.append(21, 15)  # eviction
        self.assertEqual(tester.oldest_time, 10)

    def test_get_elapsed_time(self):
        timed_buffer = TimedBuffer()
        self.assertEqual(timed_buffer.elapsed_time, 0)

        timed_buffer.append(1, 10)
        self.assertEqual(timed_buffer.elapsed_time, 0)

        timed_buffer.append(2, 12)
        self.assertEqual(timed_buffer.elapsed_time, 1)

        timed_buffer.append(7, 8)
        self.assertEqual(timed_buffer.elapsed_time, 6)

    def test_length(self):
        timed_buffer = TimedBuffer()

        self.assertEqual(len(timed_buffer), 0)

        timed_buffer.append(1, 1)
        self.assertEqual(len(timed_buffer), 1)

        timed_buffer.append(2, 2)
        timed_buffer.append(5, 10)

        self.assertEqual(len(timed_buffer), 3)

    def test_reset(self):
        timed_buffer = TimedBuffer()
        for ix in range(10):
            timed_buffer.append(ix, ix ** 2)

        self.assertEqual(len(timed_buffer), 10)

        timed_buffer.reset()

        self.assertEqual(len(timed_buffer._time_buffer), 0)
        self.assertEqual(len(timed_buffer._sample_buffer), 0)
        self.assertEqual(len(timed_buffer), 0)

    def test_change_target_time(self):
        timed_buffer = TimedBuffer(10)
        for ix in range(10):
            timed_buffer.append(ix, ix ** 2)

        timed_buffer.target_elapsed_time = 5

        self.assertEqual(len(timed_buffer), 7)
        self.assertEqual(timed_buffer.oldest_sample, 9)
        self.assertEqual(timed_buffer.newest_sample, 81)

    def testBackwardsTime(self):
        timed_buffer = TimedBuffer()

        timed_buffer.append(1, 1)

        timed_buffer.append(1, 0)
        with self.assertRaises(ValueError):
            timed_buffer.append(0, 5)


class TestTimedRunningSum(unittest.TestCase):
    def test_create(self):
        tester = TimedRunningSum(target_elapsed_time=3)

        self.assertEqual(tester._target_elapsed_time, 3)

    def test_remove_old_samples(self):
        tester = TimedRunningSum(10)

        tester._time_buffer.extend([1, 2, 13])
        tester._sample_buffer.extend([4, 5, 6])

        self.assertEqual(tester._time_buffer[0], 1)
        self.assertEqual(tester._time_buffer[1], 2)
        self.assertEqual(tester._time_buffer[2], 13)

        self.assertEqual(tester._sample_buffer[0], 4)
        self.assertEqual(tester._sample_buffer[1], 5)
        self.assertEqual(tester._sample_buffer[2], 6)

        tester.remove_old_samples()

        self.assertEqual(len(tester), 2)
        self.assertEqual(tester._time_buffer[0], 2)
        self.assertEqual(tester._sample_buffer[0], 5)

        self.assertEqual(tester._time_buffer[1], 13)
        self.assertEqual(tester._sample_buffer[1], 6)

    def test_sum_property(self):
        tester = TimedRunningSum(3)

        self.assertEqual(tester.sum, 0)

    def test_append(self):
        tester = TimedRunningSum(10.0)
        tester.append(1, 1)

        self.assertEqual(len(tester), 1)
        self.assertEqual(tester._time_buffer[0], 1)
        self.assertEqual(tester._sample_buffer[0], 1)
        self.assertEqual(tester.sum, 1)

        tester.append(2, 2)

        self.assertEqual(len(tester), 2)
        self.assertEqual(tester._time_buffer[0], 1)
        self.assertEqual(tester._sample_buffer[0], 1)
        self.assertEqual(tester._time_buffer[1], 2)
        self.assertEqual(tester._sample_buffer[1], 2)
        self.assertEqual(tester.sum, 3)

        tester.append(15, 3.14)

        # Will now have 2 elements because one older sample is kept even if it is out of range
        self.assertEqual(len(tester), 2)
        self.assertEqual(tester._time_buffer[0], 2)
        self.assertEqual(tester._sample_buffer[0], 2)
        self.assertEqual(tester._time_buffer[1], 15)
        self.assertEqual(tester._sample_buffer[1], 3.14)
        self.assertAlmostEqual(tester.sum, 5.14)

    def test_popleft(self):
        tester = TimedRunningSum(10)
        tester.append(1, 1)

        self.assertEqual(len(tester), 1)
        self.assertEqual(tester._time_buffer[0], 1)
        self.assertEqual(tester._sample_buffer[0], 1)
        self.assertAlmostEqual(tester.sum, 1)

        tester.append(2, 3.14)
        self.assertAlmostEqual(tester.sum, 4.14)

        tester.popleft()

        self.assertEqual(len(tester), 1)
        self.assertAlmostEqual(tester.sum, 3.14)

    def test_get_oldest(self):
        tester = TimedRunningSum(10)

        self.assertIsNone(tester.oldest_sample)

        tester.append(1, 2)
        self.assertEqual(tester.oldest_sample, 2)

        tester.append(10, 4)
        self.assertEqual(tester.oldest_sample, 2)

        tester.append(11.1, 15)     # No evictions yet because one older sample is kept
        self.assertEqual(tester.oldest_sample, 2)

        tester.append(11.2, 32)  # Still no eviction: 10 to 11.2 is within the window of 10
        self.assertEqual(tester.oldest_sample, 2)

        tester.append(21, 32)  # Now there should be an eviction but the (10, 4) will still be there
        self.assertEqual(tester.oldest_sample, 4)

    def test_get_newest(self):
        tester = TimedRunningSum(10)

        self.assertIsNone(tester.newest_sample)

        tester.append(1, 2)
        self.assertEqual(tester.newest_sample, 2)

        tester.append(10, 4)
        self.assertEqual(tester.newest_sample, 4)

        tester.append(11.1, 15)
        self.assertEqual(tester.newest_sample, 15)

    def test_get_last_time(self):
        tester = TimedRunningSum(10)

        self.assertEqual(tester.newest_time, 0)

        tester.append(1, 2)
        self.assertEqual(tester.newest_time, 1)

        tester.append(10, 8)
        self.assertEqual(tester.newest_time, 10)

        tester.append(11.1, 15)
        self.assertEqual(tester.newest_time, 11.1)

    def test_get_oldest_time(self):
        tester = TimedRunningSum(10)

        self.assertEqual(tester.oldest_time, 0)

        tester.append(1, 2)
        self.assertEqual(tester.oldest_time, 1)

        tester.append(10, 8)
        self.assertEqual(tester.oldest_time, 1)

        tester.append(11.1, 15)     # No eviction
        self.assertEqual(tester.oldest_time, 1)

        tester.append(21, 15)  # eviction
        self.assertEqual(tester.oldest_time, 10)

    def test_get_elapsed_time(self):
        timed_buffer = TimedRunningSum()
        self.assertEqual(timed_buffer.elapsed_time, 0)

        timed_buffer.append(1, 10)
        self.assertEqual(timed_buffer.elapsed_time, 0)

        timed_buffer.append(2, 12)
        self.assertEqual(timed_buffer.elapsed_time, 1)

        timed_buffer.append(7, 8)
        self.assertEqual(timed_buffer.elapsed_time, 6)

    def test_length(self):
        timed_buffer = TimedRunningSum()

        self.assertEqual(len(timed_buffer), 0)

        timed_buffer.append(1, 1)
        self.assertEqual(len(timed_buffer), 1)

        timed_buffer.append(2, 2)
        timed_buffer.append(5, 10)

        self.assertEqual(len(timed_buffer), 3)

    def test_reset(self):
        tester = TimedRunningSum()
        for ix in range(10):
            tester.append(ix, ix ** 2)

        self.assertEqual(len(tester), 10)

        tester.reset()

        self.assertEqual(len(tester._time_buffer), 0)
        self.assertEqual(len(tester._sample_buffer), 0)
        self.assertEqual(len(tester), 0)
        self.assertEqual(tester.sum, 0)

    def test_change_target_time(self):
        timed_buffer = TimedRunningSum(10)
        for ix in range(10):
            timed_buffer.append(ix, ix ** 2)

        timed_buffer.target_elapsed_time = 5

        self.assertEqual(len(timed_buffer), 7)
        self.assertEqual(timed_buffer.oldest_sample, 9)
        self.assertEqual(timed_buffer.newest_sample, 81)

    def testBackwardsTime(self):
        timed_buffer = TimedRunningSum()

        timed_buffer.append(1, 1)

        timed_buffer.append(1, 0)
        with self.assertRaises(ValueError):
            timed_buffer.append(0, 5)


if __name__ == '__main__':
    unittest.main()
