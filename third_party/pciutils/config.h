/*
 * Hermetic replacement for pciutils' generated lib/config.h.
 *
 * Upstream pciutils generates this header from lib/configure at build time,
 * probing the host for OS/arch/access-method support. We hand-write it so both
 * the public headers (<pci/pci.h> and friends) AND the libpci sources build
 * hermetically, with a minimal, dependency-free Linux configuration:
 *
 *   * OS / arch / ABI:
 *       PCI_OS_LINUX            -- selects the Linux code paths
 *       PCI_HAVE_64BIT_ADDRESS  -- widens pciaddr_t to 64-bit (host ABI on
 *                                  64-bit Linux)
 *       PCI_ARCH_*              -- only affects a SPARC64-specific typedef
 *   * Access methods compiled in (see lib/Makefile OBJS + lib/init.c table):
 *       PCI_HAVE_PM_LINUX_SYSFS -- /sys/bus/pci scanning (the method MORI uses)
 *       PCI_HAVE_PM_LINUX_PROC  -- /proc/bus/pci fallback
 *       PCI_HAVE_PM_DUMP        -- dump.c is always in OBJS; keep the init.c
 *                                  method table consistent with it
 *
 * Deliberately left UNDEFINED to keep the static lib free of external deps:
 *   PCI_USE_DNS (libresolv), PCI_HAVE_HWDB (libudev), PCI_COMPRESSED_IDS
 * (zlib), PCI_HAVE_PM_INTEL_CONF (iopl/ioperm), PCI_HAVE_PM_MMIO_CONF / _ECAM
 *   (physmem). MORI only calls pci_alloc/init/scan_bus/fill_info/read_byte/
 *   cleanup, so the ID-name / port / mmio machinery is never exercised.
 *
 * The arch macro is derived from the compiler's built-ins so this single
 * header works for both x86_64 and aarch64 ROCm images.
 */

#ifndef XLA_THIRD_PARTY_PCIUTILS_CONFIG_H_
#define XLA_THIRD_PARTY_PCIUTILS_CONFIG_H_

#define PCI_OS_LINUX
#define PCI_HAVE_64BIT_ADDRESS

#if defined(__x86_64__)
#define PCI_ARCH_X86_64
#elif defined(__i386__)
#define PCI_ARCH_I386
#elif defined(__aarch64__)
#define PCI_ARCH_AARCH64
#elif defined(__arm__)
#define PCI_ARCH_ARM
#elif defined(__powerpc64__)
#define PCI_ARCH_PPC64
#elif defined(__sparc__) && defined(__arch64__)
#define PCI_ARCH_SPARC64
#endif

/* Access methods (Linux). sysfs is the default/primary; proc is a fallback. */
#define PCI_HAVE_PM_LINUX_SYSFS
#define PCI_HAVE_PM_LINUX_PROC
#define PCI_HAVE_PM_DUMP

/* Filesystem paths and pci.ids location (pci.ids is never loaded here since
 * MORI does not call pci_lookup_name, so IDS_DIR is effectively cosmetic). */
#define PCI_PATH_SYS_BUS_PCI "/sys/bus/pci"
#define PCI_PATH_PROC_BUS_PCI "/proc/bus/pci"
#define PCI_PATH_IDS_DIR "/usr/share"
#define PCI_IDS "pci.ids"

#define PCILIB_VERSION "3.13.0"

#endif  // XLA_THIRD_PARTY_PCIUTILS_CONFIG_H_
