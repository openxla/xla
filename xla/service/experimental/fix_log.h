// temporary file to deal with macro overriding caused by
// including both the tsl/platform/logging.h and the 
// ortools/linear_solver/linear_solver.h
// will simply reset the VLOG* macros to that of the tsl platform
// as opposed to the absl ones that are overwritten by the ortools include

#include "tsl/platform/logging.h"


// Otherwise, set TF_CPP_MAX_VLOG_LEVEL environment to update minimum log level
// of VLOG, or TF_CPP_VMODULE to set the minimum log level for individual
// translation units.
#define VLOG_IS_ON(lvl)                                              \
  (([](int level, const char* fname) {                               \
    static const bool vmodule_activated =                            \
        ::tsl::internal::LogMessage::VmoduleActivated(fname, level); \
    return vmodule_activated;                                        \
  })(lvl, __FILE__))

#define VLOG(level)                   \
  TF_PREDICT_TRUE(!VLOG_IS_ON(level)) \
  ? (void)0                           \
  : ::tsl::internal::Voidifier() &    \
          ::tsl::internal::LogMessage(__FILE__, __LINE__, tsl::INFO)