#!/usr/bin/env python
#
# Copyright (C) 2017 Vrije Universiteit Brussel (http://vub.ac.be)
# Copyright (C) 2015 Julien Desfossez
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA  02110-1301, USA.

import cv2
import numpy as np
import json
import serial
import threading
import time
import sys
import gym
import zbar

try:
    import urllib.request as urllib
except:
    import urllib2 as urllib

from gym import spaces
from gym.utils import seeding

class LiveThread(threading.Thread):
    """ This thread constantly consumes images from the Sony camera, so that every
        action taken by the agent uses the latest (freshest) image. If images
        are read only when needed, the camera slowly buffers old images and sends
        outdated information (up to 30 seconds to 1 minute late).
    """

    def __init__(self, stream):
        threading.Thread.__init__(self)

        self._condition = threading.Condition()
        self._last_jpg = None
        self._stream = stream

    def run(self):
        data = b''

        while True:
            a = data.find(b'\xff\xd8')
            b = data.find(b'\xff\xd9')

            if a != -1 and b != -1:
                # Complete frame received
                jpg = data[a:b+2]
                data = data[b+2:]

                self._condition.acquire()
                self._last_jpg = jpg
                self._condition.notify()
                self._condition.release()

            # Need more data
            data += self._stream.read(2048)

    def get_image(self):
        self._condition.acquire()
        self._condition.wait_for(lambda: self._last_jpg is not None)

        jpg = self._last_jpg
        self._last_jpg = None   # Ensure that any image is used only once

        self._condition.release()
        return jpg

class SonyWrapper(gym.Env):
    """ Sony camera wrapper.

        This environment wraps another environment and replaces its observations
        with features detected from a video stream coming from a Sony camera.

        A Sony HDR-AS100V is used because it can stream low-resolution video on
        Wifi. Connect to the Direct-kXXX network exposed by the camera.
    """
    def __init__(self, wrapped):
        self.wrapped = wrapped

        self.url = 'http://192.168.122.1:10000/sony/camera'
        self.id = 1
        self.action_space = wrapped.action_space
        self.observation_space = spaces.Box(
            np.array([0.0] * 19),
            np.array([1.0] * 19)
        ) # 3 object detectors (one for each color), giving x, y, size, then one-hot encoding of the largest barcode visible (0 by default)

        self._scanner = zbar.Scanner()
        self._centroids = [None] * 3
        self._areas = [0.0] * 3

        # Connect to the camera
        r = self.send_basic_cmd("getVersions")
        print('Sony Camera version %s' % r['result'][0][0])

        # Enable steady mode to have a smaller field of view
        self.send_basic_cmd('setSteadyMode', ['on'])

        # Open liveview stream
        self.send_basic_cmd('stopLiveview')
        r = self.send_basic_cmd('startLiveview')
        self._stream = urllib.urlopen(r['result'][0])

        self._thread = LiveThread(self._stream)
        self._thread.start()

        print('Liveview stream opened')

    def render(self, **kwargs):
        return self.wrapped.render(**kwargs)

    def _step(self, action):
        # Execute the action in the wrapped environment, then add the observation
        # from the camera
        _, reward, done, info = self.wrapped._step(action)

        return self.observe(), 0.0, False, info

    def _reset(self):
        self.wrapped._reset()

        return self.observe()

    def observe(self):
        try:
            return self.process_image(self._thread.get_image())
        except:
            # Bad image, try next one
            return self.observe()

    def detect_color(self, frame_hsv, h_low, h_high, index):
        sel = cv2.inRange(frame_hsv, h_low, h_high)

        # Remove some noise using a closing operation
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        sel = cv2.erode(sel, element, iterations=1)
        sel = cv2.dilate(sel, element, iterations=2)
        sel = cv2.erode(sel, element, iterations=1)

        # Update the centroid for this color
        rs = cv2.connectedComponentsWithStats(sel, 8)
        largest_center = None
        largest_area = 20       # Ignore small objects

        for i in range(1, rs[0]):
            area = rs[2][i, cv2.CC_STAT_AREA]
            center = rs[3][i]

            if area > largest_area:
                largest_area = area
                largest_center = center

        self._centroids[index] = largest_center     # May be None if the object is out of frame
        self._areas[index] = 0.0 if largest_center is None else largest_area

    def process_image(self, data):
        frame = cv2.imdecode(np.fromstring(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_plane = frame_hsv[:, :, 0]
        h_plane[h_plane <= 2] = 190

        # Detect the largest red, green and blue objects
        self.detect_color(frame_hsv, (175, 150, 100), (190, 255, 255), 0)
        self.detect_color(frame_hsv, (60, 128, 90), (72, 255, 255), 1)
        self.detect_color(frame_hsv, (105, 128, 100), (122, 255, 255), 2)

        # Make observation
        observation = np.zeros((9+10,), dtype=np.float32)

        for i in range(3):
            if self._centroids[i] is not None:
                cx, cy = self._centroids[i]
                size = self._areas[i]

                observation[i*3 + 0] = cx / frame.shape[1]
                observation[i*3 + 1] = cy / frame.shape[0]
                observation[i*3 + 2] = size / (frame.shape[0] * frame.shape[1])

                color = [(0, 0, 255), (0, 255, 0), (255, 0, 0)][i]
                size = int(np.sqrt(size))
                cx, cy = int(cx), int(cy)
                cv2.rectangle(frame, (cx-size, cy-size), (cx+size, cy+size), color, 3)

        # Observe bar codes
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        symbols = self._scanner.scan(frame_gray)
        max_size = 0
        max_data = None

        for s in symbols:
            width = max(p[0] for p in s.position) - min(p[0] for p in s.position)
            height = max(p[1] for p in s.position) - min(p[1] for p in s.position)
            size = width * height

            cv2.rectangle(frame, s.position[3], s.position[1], (255, 255, 255), 3)

            if size > max_size:
                if len(s.data) == 1 and s.data.isdigit():
                    max_data = int(s.data)
                    max_size = size

        if max_data is not None:
            observation[9 + max_data] = 1.0

        cv2.imshow('image', frame)
        cv2.waitKey(1)

        return observation

    def send_rq(self, data):
        req = urllib.Request(self.url)
        req.add_header('Content-Type', 'application/json')
        data["id"] = self.id
        self.id += 1
        response = urllib.urlopen(req, bytes(json.dumps(data), 'utf-8'))
        r = json.loads(str(response.read(), 'utf-8'))
        return r

    def send_basic_cmd(self, cmd, params=[]):
        data = {"method": cmd,
                "params": params,
                "version": "1.0"}
        return self.send_rq(data)

if __name__ == '__main__':
    # Test image processing
    env = SonyWrapper(None)

    f = open(sys.argv[1], 'rb')
    observation = env.process_image(f.read())
    print(observation)
