//
//  timer.hpp
//  fit3dcicle
//
//  Created by xiao.hu on 2023/8/21.
//  Copyright © 2023 xiao.hu. All rights reserved.
//

#ifndef timer_h
#define timer_h

#include <iostream>
#include <chrono>
#include <string>

class Timer{
public:
    Timer():t1(res::zero()),t2(res::zero())
    {
        tic();
    }
    ~Timer(){}
    void tic()
    {
        t1 = clock::now();
    }
    void toc(const std::string &str)
    {
        t2 = clock::now();
        std::cout << str << " time: " << std::chrono::duration_cast<res>(t2-t1).count()/1e3 << " ms. " << std::endl;
    }
private:
    typedef std::chrono::high_resolution_clock clock;
    typedef std::chrono::microseconds res;
    clock::time_point t1;
    clock::time_point t2;
};


#endif /* timer_h */
