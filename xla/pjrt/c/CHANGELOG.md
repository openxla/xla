# PJRT C API changelog

## 1.0 (Oct 17, 2023)

* Bumping version to 1.0 to signify beginning of ABI compatibility guarantees
  (although we are unofficially best-effort honoring previous changes as
  well). See
  https://docs.google.com/document/d/1TKB5NyGtdzrpgw5mpyFjVAhJjpSNdF31T6pjPl_UT2o/edit
  for ABI compatibility details. No other changes.

## 0.35 (Oct 17, 2023)

* Added const to a bunch of lists and value types.

## 0.34 (Oct 9, 2023)

* Added PJRT_Structure_Type::PJRT_Structure_Type_Profiler.

## 0.33 (Oct 3, 2023)

* Added PJRT_Client_CreateViewOfDeviceBuffer.

## 0.32 (Sep 26, 2023)

* Added PJRT_Buffer_CopyToMemory.

## 0.31 (Sep 22, 2023)

* Added PJRT_Structure_Base.
* Added PJRT_Structure_Type.
* Renamed PJRT_Api.priv to PJRT_Api.extension_start.

## 0.30 (Sep 14, 2023)

* Added PJRT_NamedValue_Type::PJRT_NamedValue_kBool.

## 0.29 (Sep 6, 2023)

* Added PJRT_Executable_OutputElementTypes.
* Added PJRT_Executable_OutputDimensions.
