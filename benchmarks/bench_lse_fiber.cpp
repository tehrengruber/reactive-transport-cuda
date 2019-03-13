#include <boost/fiber/all.hpp>
#include <Eigen/Dense>

using numeric_t = double;

using namespace Eigen;

struct SimpleTaskAggregator {
    using cv_t = boost::fibers::condition_variable;
    using mutex_t = boost::fibers::mutex;

    static constexpr size_t N = 28;

    size_t bin_size = 10;

    size_t num_bins = 10;

    size_t current_bin = 0;
    size_t current_bin_idx = 10;

    std::vector<size_t> free_bins;

    mutex_t mutex;

    cv_t* cvs;
    mutex_t* mutexes;

    cv_t free_cvs;

    numeric_t* As;
    numeric_t* bs;

    SimpleTaskAggregator() {
        cvs = new cv_t[num_bins]();
        mutexes = new mutex_t[num_bins]();
        As = new numeric_t[num_bins*bin_size*N*N];
        bs = new numeric_t[num_bins*bin_size*N];

        for (size_t i=1; i<num_bins; ++i) {
            free_bins.push_back(i);
        }
    }

    ~SimpleTaskAggregator() {
        delete[] cvs;
        delete[] As;
        delete[] bs;
        delete[] mutexes;
    }

    SimpleTaskAggregator(const SimpleTaskAggregator&) = delete;
    SimpleTaskAggregator& operator=(const SimpleTaskAggregator&) = delete;

    Eigen::Matrix<numeric_t, N, 1> solve(Eigen::Matrix<numeric_t, N, N>& A, Eigen::Matrix<numeric_t, N, 1>& b) {
        // reserve slot in bin
        if (current_bin_idx==bin_size) {
            process_current_bin();

            // if there are no free bins available wait until one bin finishes
            while (free_bins.size() == 0) {
                std::unique_lock<mutex_t> lock(mutex);
                free_cvs.wait(lock);
            }

            current_bin = free_bins.back();
            std::cout << "new bin:" << current_bin << std::endl;
            free_bins.pop_back();
            current_bin_idx = 0;
            //boost::this_fiber::yield();
            //return Eigen::Matrix<numeric_t, N, 1>();
        }

        // copy data
        size_t idx = current_bin*bin_size+current_bin_idx;
        for (size_t i=0; i<N*N; ++i) {
            As[idx*N*N+i] = *(A.data()+i);
        }
        for (size_t i=0; i<N*N; ++i) {
            bs[idx*N+i] = *(b.data()+i);
        }

        current_bin_idx++;

        std::cout << "wait: " << current_bin << " " << current_bin_idx << std::endl;

        //boost::this_fiber::yield();
        // wait until the bin has been processed
        {
            std::unique_lock<mutex_t> lock(mutexes[current_bin]);
            cvs[current_bin].wait(lock);
        }

        std::cout << "wake up" << std::endl;

        // copy result
        Eigen::Matrix<numeric_t, N, 1> x(bs+idx*N);
        return x;
    }

    void process_current_bin() {
        std::cout << "process" << std::endl;
        size_t bin = current_bin;
        size_t bin_occupancy = current_bin_idx-1;
        //assert(bin_occupancy!=0);

        for (size_t i=0; i<=bin_occupancy; ++i) {
            Map<Eigen::Matrix<numeric_t, N, N>> A(As+bin*N*N*i);
            Map<Eigen::Matrix<numeric_t, N, 1>> b(bs+bin*N*i);
            b = A.partialPivLu().solve(b);
        }

        // release bin
        free_bins.push_back(bin);

        std::cout << "notify: " << bin << std::endl;

        // notify
        cvs[bin].notify_all();
    }
};

SimpleTaskAggregator task_aggregator;

int fiber_count = 0;
boost::fibers::condition_variable cv;

boost::fibers::mutex m;

class custom_scheduler : public boost::fibers::algo::round_robin {
    virtual boost::fibers::context * pick_next() noexcept {
        boost::fibers::context* victim = boost::fibers::algo::round_robin::pick_next();
        if (victim == nullptr) {
            std::cout << "empty" << std::endl;
            task_aggregator.process_current_bin();
            victim = boost::fibers::algo::round_robin::pick_next();
        }
        //std::cout << "schedule" << std::endl;
        return victim;
    }
};

void fiber() {
    for (size_t i=0; i<1; ++i) {
        Eigen::Matrix<numeric_t, 28, 28> A;
        A.setRandom();
        A.diagonal().setConstant(10);
        Eigen::Matrix<numeric_t, 28, 1> b;
        b.setRandom();

        auto x = task_aggregator.solve(A, b);

        std::cout << (A*x-b).norm() << std::endl;
    }
}

void main_fiber() {
    for (size_t i=0; i<11; ++i) {
        boost::fibers::fiber(fiber).detach();
        fiber_count++;
    }
}

int main() {
    boost::fibers::use_scheduling_algorithm< custom_scheduler >();

    boost::fibers::fiber mf(main_fiber);
    mf.join();
}