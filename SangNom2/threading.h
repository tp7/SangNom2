#ifndef __THREADING_H__
#define __THREADING_H__

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

//inspired by http://progsch.net/wordpress/?p=81
class ThreadPool {
public:
    explicit ThreadPool(int threads): quit(false), runningThreads(0) {
        for (int i = 0; i < threads; ++i) {
            this->threads.emplace_back([&]{
                while(true) {
                    std::unique_lock<std::mutex> lock(this->mutex);
                    while(!this->quit && this->tasks.empty()) {
                        this->internalCondition.wait(lock);
                    }
                    if (this->quit) {
                        return;
                    }
                    auto function = this->tasks.front();
                    this->tasks.pop();
                    ++runningThreads;
                    lock.unlock();

                    function();
                    
                    lock.lock();
                    --runningThreads;
                    lock.unlock();

                    externalCondition.notify_all();
                }
            });
        }
    }

    template<class T>
    void enqueue(T &&function) {
        {
            std::unique_lock<std::mutex> lock(mutex);
            tasks.push(function);
        }
        internalCondition.notify_one();
    }

    ~ThreadPool() {
        askToQuit();
        for(auto &thread: threads) {
            thread.join();
        }
    }

    void waitAll() {
        std::unique_lock<std::mutex> lock(this->mutex);
        while(!this->tasks.empty() || runningThreads > 0) {
            this->externalCondition.wait(lock);
        }
    }

    void askToQuit() {
        {
            std::unique_lock<std::mutex> lock(this->mutex);
            this->quit = true;
        }
        
        internalCondition.notify_all();
        externalCondition.notify_all();
    }

    int numberOfThreads() {
        return threads.size();
    }

private:
    std::vector<std::thread> threads;
    std::queue<std::function<void()>> tasks;

    std::mutex mutex;
    std::condition_variable internalCondition;
    std::condition_variable externalCondition;
    bool quit;
    int runningThreads;
};



#endif