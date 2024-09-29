package main

/*
#include <stdio.h>
#include <stdlib.h>
#cgo CXXFLAGS: -std=c++23 -I. -I${SRCDIR}/../src -I/usr/local/cuda-12.4/targets/x86_64-linux/include
#cgo LDFLAGS: -L${SRCDIR} -lengine_lib -L/usr/local/cuda-12.4/targets/x86_64-linux/lib -lcudart -Wl,-rpath,${SRCDIR}:/usr/local/cuda-12.4/targets/x86_64-linux/lib
#include "inference_wrapper.h"
*/
import "C"

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"runtime"
	"sync"
	"unsafe"
)

var (
	session      *C.InferenceSessionWrapper
	sessionMutex sync.Mutex
)

type InferenceRequest struct {
	Data []float32 `json:"data"`
}

type InferenceResponse struct {
	Result []float32 `json:"result"`
}

func initSession(configPath string) error {
	sessionMutex.Lock()
	defer sessionMutex.Unlock()

	cConfigPath := C.CString(configPath)
	defer C.free(unsafe.Pointer(cConfigPath))

	session = C.create_session(cConfigPath)
	if session == nil {
		return fmt.Errorf("failed to create inference session")
	}

	result := C.initialize_provider(session)
	if result != 0 {
		C.destroy_session(session)
		return fmt.Errorf("failed to initialize provider")
	}

	return nil
}

func cleanupSession() {
	sessionMutex.Lock()
	defer sessionMutex.Unlock()

	if session != nil {
		C.destroy_session(session)
		session = nil
	}
}

func inferenceHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req InferenceRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if len(req.Data) == 0 {
		http.Error(w, "Input data is empty", http.StatusBadRequest)
		return
	}

	cData := (*C.float)(unsafe.Pointer(&req.Data[0]))
	cSize := C.uint64_t(len(req.Data))

	sessionMutex.Lock()
	cResult := C.run_inference(session, cData, cSize)
	sessionMutex.Unlock()
	defer C.free_result(cResult)

	resultSlice := (*[1 << 30]C.float)(unsafe.Pointer(cResult.data))[:cResult.size:cResult.size]
	goResult := make([]float32, cResult.size)
	for i, v := range resultSlice {
		goResult[i] = float32(v)
	}

	response := InferenceResponse{Result: goResult}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func main() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	configPath := "/home/michal/code/inference_engine/model_repository/mnist.yaml"
	if err := initSession(configPath); err != nil {
		log.Fatalf("Failed to initialize inference session: %v", err)
	}
	defer cleanupSession()

	http.HandleFunc("/infer", inferenceHandler)

	log.Println("Server starting on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
