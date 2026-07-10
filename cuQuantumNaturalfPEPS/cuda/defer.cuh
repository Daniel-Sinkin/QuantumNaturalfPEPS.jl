#pragma once
#include <utility>

namespace qnpeps
{
template <typename F>
class Defer
{
  public:
    explicit Defer(F fn) : fn_{std::move(fn)} {}
    ~Defer() noexcept { fn_(); }

    Defer(const Defer&) = delete;
    Defer(Defer&&) = delete;
    auto operator=(const Defer&) -> Defer& = delete;
    auto operator=(Defer&&) -> Defer& = delete;

  private:
    F fn_;
};
template <typename F>
Defer(F) -> Defer<F>;
}

#define QN_DEFER_CONCAT_(a, b) a##b
#define QN_DEFER_CONCAT(a, b) QN_DEFER_CONCAT_(a, b)
#define DEFER(...)                                                                                 \
    const qnpeps::Defer QN_DEFER_CONCAT(qnpeps_defer_, __LINE__)                                   \
    {                                                                                              \
        __VA_ARGS__                                                                                \
    }
