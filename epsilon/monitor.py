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

import abc
import json
import logging


class Monitor(abc.ABC):
    """
    @brief A base class for GNSS monitors.

    This abstract base class defines the monitor structure and provides basic functionality to be used by subclasses.
    """

    def __init__(self, receiver_id=None, monitor_timeout=None, threshold=0.0):
        """
        Create a new monitor

        @param receiver_id An identifier for the source of data this monitor is watching
        @param monitor_timeout If this much time passes before another message comes in, reset the monitor; None
               indicates no timeout
        @param threshold The threshold above which monitor metrics will trigger alarms
        """

        self.logger = logging.getLogger(self.__class__.__name__)
        self._last_event_time = None

        self.receiver_id = receiver_id
        self.monitor_timeout = monitor_timeout
        self._threshold = 0

        # Status variables
        self._status = {'alarm': False, 'threshold': self._threshold, 'metric': None}

        self.threshold = threshold  # Force error checking

        self._id_str = '{} for {}'.format(self.__class__.__name__, self.receiver_id)
        logging.debug('Created monitor %s with a timeout of %s', self._id_str, str(self.monitor_timeout))

    @property
    def threshold(self):
        """
        @brief The threshold for the monitor; definition will vary from subclass to subclass
        """

        return self._threshold

    @threshold.setter
    def threshold(self, threshold):
        """
        Set the threshold; subclasses may override this to change valid ranges and error handling

        @param threshold: The new threshold value.
        """

        if threshold is None:
            raise ValueError('Monitor threshold must be defined but it was set to None!')

        self._threshold = threshold
        self._status['threshold'] = threshold

    @property
    def metric(self):
        return self._status['metric']

    @property
    def alarm(self):
        return self._status['alarm']

    @abc.abstractmethod
    def _calculate_metric(self, message):
        """
        Process the given message and return the monitor metric; this MUST be overridden in subclasses.

        @param message: The message to process.
        @return: The metric value.
        """

        pass

    def _compare_metric(self, metric):
        """
        Compare the metric to this monitor's threshold and return True if it is abnormal and False otherwise.
        Subclasses may override this.

        @param metric: The new metric.
        @return: True if the comparison is abnormal and False otherwise.
        """

        return metric > self._threshold

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

        if message['receiver_id'] != self.receiver_id:
            self.logger.debug('Message is from a different receiver %s, instead of %s',
                              message['receiver_id'],
                              self.receiver_id)

            return False

        if self._last_event_time is not None and message['rxTime'] < self._last_event_time:
            self.logger.warning('Received a message out of order. Time is %s but last time was %s',
                                message['rxTime'],
                                self._last_event_time)
            return False

        return True

    def update(self, message):
        """
        Update the monitor with new data, resetting first if too much time has passed between events.

        This base class method DOES NOT set the _alarm variable.

        @param message A dictionary containing the fields this monitor needs.

        @return True if there is an alarm, False if there isn't, and None if a metric could not be computed.
        """

        if not self.verify_message(message):
            return None

        self.logger.debug('Updating %s', self._id_str)

        # This check is here because if it fails the monitor is reset, unlike verify_message which only determines
        #  whether the message should be read at all
        if self.monitor_timeout is not None and self._last_event_time is not None:
            if message['rxTime'] - self._last_event_time >= self.monitor_timeout:
                self.reset()

        self._last_event_time = message['rxTime']

        metric = self._calculate_metric(message)

        if metric is None:
            return None

        alarm = self._compare_metric(metric)

        self._status['metric'] = metric
        self._status['alarm'] = alarm

        return alarm

    def reset(self):
        """
        Reset the monitor.
        """

        self.logger.info('Resetting %s', self._id_str)
        self._last_event_time = None
        self._status['alarm'] = False
        self._status['metric'] = None

    def get_status(self):
        """
        Get a json string/message containing the monitor's status.

        The fields are defined by each subclass, though all contain "alarm" as defined in this class.

        @return: a json representation of this monitor's status.
        """

        return json.dumps(self._status)


def from_config(monitor_name, configuration, cls=Monitor):
    """
    Create a monitor from the given configuration by checking if the requested monitor exists as a subclass
    of Monitor and returning a new instance of that subclass if so. NOTE: Only children and grandchildren of Monitor
    may be created this way.

    @param monitor_name The name of the monitor (that is, subclass) to instantiate.
    @param configuration A dictionary containing arguments for the particular monitor
    @param cls Used for recursion; parent to check for usable subclasses next.

    @return The created monitor, or None if creation was not possible.
    """

    # Using logging here since there is no other logger defined in this scope
    logging.info('Creating monitor from configuration')
    logging.debug('Config is %s', configuration)

    # Though this method is abstract, the base class will call the correct subclass' version of this method
    # allowing this method to act as a Monitor factory

    logging.debug('Looking for a monitor for %s', monitor_name)

    # See if there's a child or grandchild of Monitor that matches the given name
    for sub in cls.__subclasses__():
        if monitor_name == sub.__name__:
            logging.debug('Found monitor %s', monitor_name)
            return sub(**configuration)

    # Recurse if necessary
    for sub in cls.__subclasses__():
        return from_config(monitor_name, configuration, cls=sub)

    # Getting here means no suitable class was found
    if cls == Monitor:
        logging.error('Could not create %s; there is no implementation', monitor_name)

    return None
