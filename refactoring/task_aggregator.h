#ifndef REACTIVETRANSPORTGPU_TASK_AGGREGATOR_H
#define REACTIVETRANSPORTGPU_TASK_AGGREGATOR_H

struct TaskProcessorProto {
    void process();
};


auto task = [] (numeric_t T, numeric_t P, ) {

};

template <typename TASKPROCESSOR, typename INPUTQUE, typename OUTPUTQUE>
struct TaskAggregator {
    QueueManager queue_manager;

    size_t input_queue_length = 0;

    size_t min_batch_size = 1<<5;

    std::condition_variable work_avail_cv;

    std::mutex processor_running_m;

    // for each bin we store the number of results still to be consumed
    std::vector<size_t> outstanding_release_ops;

    // a mutex used by the consumer
    boost::fibers::mutex queue_m;

    // for each bin we have a condition variable on which the consumers wait until the output is ready
    std::vector<boost::fibers::condition_variable> consumer_cvs;

    // a list of bins for which processing has finished, but which have not been released by the consumers
    std::list<size_t> to_be_freed;

    TaskAggregator() {
        // todo make sure min_batch_size is a power of two
        boost:fibers::fiber([this] () {
            std::unique_lock<std::mutex> lk(processor_running_m);
            while (true) {
                // wait until work is available (during waiting processor_running_m is released)
                cv.wait(lk);
                // process
                batch_ticket_t t = queue_manager.batch_deque(min_batch_size);
                process(t.front(), t.back());
            }
        });
    }

    /*
     * Acquire a buffer in the input que and return a ticket to access it
     */
    ticket_t acquire_request_buffer() {
        if (!input_queue_manager.has_free_slots()) {
            std::unique_lock<std::mutex> lk(queue_m);
            free_slots_cv.wait(lk, [this](){ return input_queue_manager.has_free_slots(); });
        }

        return queue_manager.acquire();
    }

    size_t _get_batch_id_from_ticket(ticket_t& t) {
        return t.id >> 5;
    }

    size_t _get_first_in_batch(size_t batch_id) {
        return batch_id << 5;
    }

    /*
     * Submit input to the task processor
     * PRE: t is a valid ticket of the input que
     * POST:
     */
    void submit(ticket_t t) {
        input_queue_manager.produce(t);

        // notify the task processor that there are tasks to be processed
        if (queue_manager.length() > batch_size) {
            std::unique_lock<std::mutex> lock(processor_running_m, std::defer_lock);
            // try to acquire the lock
            if (lock.try_lock()) {
                work_avail_cv.notify_all();
            }
        }

        // wait until the task has been processed
        size_t batch_id = _get_batch_id_from_ticket(t);
        outstanding_release_ops[batch_id]++;

        std::lock_guard<std::mutex> lock(consumer_m);
        consumer_cvs[batch_id].wait(consumer_m);
    }

    void notify_recieved(ticket_t t) {
        size_t batch_id = _get_batch_id_from_ticket(t);
        num_waiters[batch_id]--;
    }
};

struct StreamLUSolver : TaskAggregator {
    OverlappingBatchLUSolver lu_solver;

    LSE input(ticket_t t) {

    }

    Vector<numeric_t, N>& output(ticket_t t) {
        _wait_for_output(t);
    }
};

void usage_submitter() {
    MessageAggregator aggregator();

    // acquire a slot in the input queue
    input_ticket_t input_ticket = aggregator.acquire_request_buffer();
    LSE& lse = aggregator.input(ticket);

    // assemble linear system of equations
    // ...

    // submit input
    output_ticket_t output_ticket = aggregator.submit(input_ticket);

    // obtain result
    aggregator.output(output_ticket);

    aggregator.notify_recieved(output_ticket);
}

/*
 *
 */
#endif //REACTIVETRANSPORTGPU_TASK_AGGREGATOR_H
