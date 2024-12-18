#include "../include/vision_transformer.h"
#include "../include/utils.h"

#include <iostream>
#include <chrono>
#include <openacc.h>

#include <nvtx3/nvToolsExt.h>

using namespace std;

int main(int argc, char *argv[]) {
    acc_init(acc_device_default);

    nvtxRangePushA("if");
    if (argc == 5) {
        string cvit_path = argv[1];
        string cpic_path = argv[2];
        string cprd_path = argv[3];
        string measure_path = argv[4];

        VisionTransformer vit;
        auto start_time = chrono::high_resolution_clock::now();
        nvtxRangePushA("VisionTransformer");
        load_cvit(cvit_path, vit);
        nvtxRangePop();
	auto end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> load_cvit_time = end_time - start_time;
    
        PictureBatch pic;
        start_time = chrono::high_resolution_clock::now();
	nvtxRangePushA("PictureBatch");
        load_cpic(cpic_path, pic);
	nvtxRangePop();
        end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> load_cpic_time = end_time - start_time;

        PredictionBatch pred;

        RowVector times;
	nvtxRangePushA("timed_forward");

	// Ensure data is moved to GPU before processing
//	#pragma acc enter data copyin(vit, pic)
        vit.timed_forward(pic, pred, times);
//	#pragma acc exit data copyout(pred, times) delete(vit, pic)
	nvtxRangePop();

        start_time = chrono::high_resolution_clock::now();
        store_cprd(cprd_path, pred);
        end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> store_cprd_time = end_time - start_time;

        ofstream measure_file(measure_path.c_str(), ios::app);
        measure_file << pred.get_B() << ";" << vit.get_depth() << ";";
        measure_file << load_cvit_time.count() << ";" << load_cpic_time.count() << ";";
        for (int i=0;i<times.get_DIM();++i) {
            measure_file << times.at(i) << ";";
        }
        measure_file << store_cprd_time.count() << endl;
        measure_file.close();

    } else {
        cout << "Usage: vit <cvit_path> <cpic_path> <cprd_path> <measure_file_path>" << endl;
    }
    nvtxRangePop();

    acc_shutdown(acc_device_default);
    return 0;
}
