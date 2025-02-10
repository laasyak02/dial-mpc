#include <functional>

class Simulation
{
public:
    Simulation(std::function<void(void)> loop_func, std::function<void(void)> exit_func = []() {}) : loop_func_(loop_func), exit_func_(exit_func)
    {
    }

    Simulation(const Simulation &) = delete;

    Simulation &operator=(const Simulation &) = delete;

    virtual ~Simulation() = default;

    virtual void Initialize(const std::string &model_path)
    {
    }

    virtual void Configure(void *config)
    {
    }

    virtual void Finalize()
    {
    }

    void Loop()
    {
        while (true)
        {
            LoopPrologue();
            loop_func_();
            LoopEpilogue();
        }
    }

    virtual void Exit()
    {
        exit_func_();
    }

protected:
    virtual void LoopPrologue() {}

    virtual void LoopEpilogue() {}

    std::function<void(void)> loop_func_;
    std::function<void(void)> exit_func_;
};