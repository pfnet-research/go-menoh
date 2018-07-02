# external package

## [Windows] Develop bind to DLL

**for developers**

Binding file `functions_windows.go` is made by `mksyscall_windows.go`.

```bash
$ cd external
$ go generate  # functions_windows.go will be created
```

However, some code is modified manually

### DLL loader

Auto-generated code is:

```go
modmenoh = windows.NewLazySystemDLL("menoh.dll")
```

Modified:

```go
modmenoh = syscall.NewLazyDLL("menoh.dll")
```

### Error handler

Auto-generated code is:

```go
r1, _, e1 := syscall.Syscall(procmenoh_make_model_data_from_onnx.Addr(), 2, uintptr(unsafe.Pointer(path)), uintptr(mdHandle), 0)
if r1 == 0 {
	if e1 != 0 {
		err = errnoErr(e1)
	} else {
		err = syscall.EINVAL
	}
}
```

Modified:

```go
r1, _, _ := syscall.Syscall(procmenoh_make_model_data_from_onnx.Addr(), 2, uintptr(unsafe.Pointer(path)), uintptr(mdHandle), 0)
if r1 != 0 {
	err = errnoErr(syscall.Errno(r1))
}
```

Other codes call `Syscall` are same to be modified.
