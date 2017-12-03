/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <chrono>
#include <iostream>
#include <thread>

#include "caffe2/core/timer.h"
#include <gtest/gtest.h>

namespace caffe2 {
namespace {

TEST(TimerTest, Test) {
  Timer timer;
  // A timer auto-starts when it is constructed.
  EXPECT_GT(timer.NanoSeconds(), 0);
  // Sleep for a while, and get the time.
  timer.Start();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  float ns = timer.NanoSeconds();
  float us = timer.MicroSeconds();
  float ms = timer.MilliSeconds();
  // Time should be at least accurate +- 10%.
  EXPECT_NEAR(ns, 100000000, 10000000);
  EXPECT_NEAR(us, 100000, 10000);
  EXPECT_NEAR(ms, 100, 10);
  // Test restarting the clock.
  timer.Start();
  EXPECT_LT(timer.NanoSeconds(), 1000);
}

TEST(TimerTest, TestLatency) {
  constexpr int iter = 1000;
  float latency = 0;
  Timer timer;
  for (int i = 0; i < iter; ++i) {
    timer.Start();
    latency += timer.NanoSeconds();
  }
  std::cout << "Average nanosecond latency is: " << latency / iter << std::endl;
  latency = 0;
  for (int i = 0; i < iter; ++i) {
    timer.Start();
    latency += timer.MicroSeconds();
  }
  std::cout << "Average microsecond latency is: " << latency / iter << std::endl;
  latency = 0;
  for (int i = 0; i < iter; ++i) {
    timer.Start();
    latency += timer.MilliSeconds();
  }
  std::cout << "Average millisecond latency is: " << latency / iter << std::endl;
}

}  // namespace
}  // namespace caffe2
