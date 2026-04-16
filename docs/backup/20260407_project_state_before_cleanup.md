# Project State Before Cleanup (2026-04-07)

## Git Status
```
On branch eval/real-ulip-benchmark
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	modified:   .gitignore
	modified:   README.md
	modified:   ULIP/AI_ETHICS.md
	modified:   ULIP/CODEOWNERS
	modified:   ULIP/CODE_OF_CONDUCT.md
	modified:   ULIP/CONTRIBUTING-ARCHIVED.md
	modified:   ULIP/CONTRIBUTING.md
	modified:   ULIP/LICENSE.txt
	modified:   ULIP/README.md
	modified:   ULIP/SECURITY.md
	modified:   ULIP/assets/figure2_resize.gif
	modified:   ULIP/assets/pipeline_8s_timing.gif
	modified:   ULIP/data/ModelNet40.yaml
	modified:   ULIP/data/Objaverse_Lvis_Colored.yaml
	modified:   ULIP/data/ShapeNet-55.yaml
	modified:   ULIP/data/dataset_3d.py
	modified:   ULIP/data/dataset_catalog.json
	modified:   ULIP/data/labels.json
	modified:   ULIP/data/templates.json
	modified:   ULIP/main.py
	modified:   ULIP/models/ULIP_models.py
	modified:   ULIP/models/customized_backbone/customized_backbone.py
	modified:   ULIP/models/losses.py
	modified:   ULIP/models/pointbert/PointTransformer_8192point.yaml
	modified:   ULIP/models/pointbert/ULIP_2_PointBERT_10k_colored_pointclouds.yaml
	modified:   ULIP/models/pointbert/checkpoint.py
	modified:   ULIP/models/pointbert/dvae.py
	modified:   ULIP/models/pointbert/logger.py
	modified:   ULIP/models/pointbert/misc.py
	modified:   ULIP/models/pointbert/point_encoder.py
	modified:   ULIP/models/pointmlp/pointMLP.py
	modified:   ULIP/models/pointnet2/pointnet2.py
	modified:   ULIP/models/pointnet2/pointnet2_utils.py
	modified:   ULIP/models/pointnext/PointNeXt/.github/workflows/main.yml
	modified:   ULIP/models/pointnext/PointNeXt/.gitignore
	modified:   ULIP/models/pointnext/PointNeXt/.gitmodules
	modified:   ULIP/models/pointnext/PointNeXt/LICENSE
	modified:   ULIP/models/pointnext/PointNeXt/README.md
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/default.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/modelnet40ply2048/assanet-l.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/modelnet40ply2048/deepgcn.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/modelnet40ply2048/default.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/modelnet40ply2048/pix4point.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/modelnet40ply2048/pointmlp.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/modelnet40ply2048/pointnet++.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/modelnet40ply2048/pointnet++_original.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/modelnet40ply2048/pointnext-s.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/s3dis/assanet-l.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/s3dis/assanet.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/s3dis/deepgcn.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/s3dis/default.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/s3dis/dgcnn.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/s3dis/pointnet++.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/s3dis/pointnet++msg.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/s3dis/pointnet.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/s3dis/pointnext-b.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/s3dis/pointnext-l.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/s3dis/pointnext-s.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/s3dis/pointnext-xl.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/s3dis_pix4point/default.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/s3dis_pix4point/pix4point.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/s3dis_pix4point/pix4point_bert.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/scannet/ASSANet.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/scannet/default.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/scannet/pointnet++.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/scannet/pointnet++_original.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/scannet/pointnext-b.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/scannet/pointnext-l.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/scannet/pointnext-s.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/scannet/pointnext-xl.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/scanobjectnn/default.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/scanobjectnn/dgcnn.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/scanobjectnn/pointmlp.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/scanobjectnn/pointnet++.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/scanobjectnn/pointnet.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/scanobjectnn/pointnext-s.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/scanobjectnn_pix4point/default.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/scanobjectnn_pix4point/pix4point.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/shapenetpart/default.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/shapenetpart/pointnext-s.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/shapenetpart/pointnext-s_c160.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/shapenetpart/pointnext-s_c64.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/shapenetpart_pix4point/default.yaml
	modified:   ULIP/models/pointnext/PointNeXt/cfgs/shapenetpart_pix4point/pix4point.yaml
	modified:   ULIP/models/pointnext/PointNeXt/docs/changes.md
	modified:   ULIP/models/pointnext/PointNeXt/docs/examples/modelnet.md
	modified:   ULIP/models/pointnext/PointNeXt/docs/examples/s3dis.md
	modified:   ULIP/models/pointnext/PointNeXt/docs/examples/scannet.md
	modified:   ULIP/models/pointnext/PointNeXt/docs/examples/scanobjectnn.md
	modified:   ULIP/models/pointnext/PointNeXt/docs/examples/shapenetpart.md
	modified:   ULIP/models/pointnext/PointNeXt/docs/index.md
	modified:   ULIP/models/pointnext/PointNeXt/docs/modelzoo.md
	modified:   ULIP/models/pointnext/PointNeXt/docs/misc/wandb.png
	modified:   ULIP/models/pointnext/PointNeXt/docs/projects/misc/effects_training_scaling.png
	modified:   ULIP/models/pointnext/PointNeXt/docs/projects/misc/pix4point.png
	modified:   ULIP/models/pointnext/PointNeXt/docs/projects/misc/pointnext.jpeg
	modified:   ULIP/models/pointnext/PointNeXt/docs/projects/misc/s3dis_vis.png
	modified:   ULIP/models/pointnext/PointNeXt/docs/projects/misc/shapenetpart_vis.png
	modified:   ULIP/models/pointnext/PointNeXt/docs/projects/pix4point.md
	modified:   ULIP/models/pointnext/PointNeXt/docs/projects/pointnext.md
	modified:   ULIP/models/pointnext/PointNeXt/examples/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/examples/classification/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/examples/classification/main.py
	modified:   ULIP/models/pointnext/PointNeXt/examples/classification/pretrain.py
	modified:   ULIP/models/pointnext/PointNeXt/examples/classification/train.py
	modified:   ULIP/models/pointnext/PointNeXt/examples/profile.py
	modified:   ULIP/models/pointnext/PointNeXt/examples/segmentation/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/examples/segmentation/main.py
	modified:   ULIP/models/pointnext/PointNeXt/examples/segmentation/test_s3dis_6fold.py
	modified:   ULIP/models/pointnext/PointNeXt/examples/segmentation/vis_results.py
	modified:   ULIP/models/pointnext/PointNeXt/examples/shapenetpart/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/examples/shapenetpart/main.py
	modified:   ULIP/models/pointnext/PointNeXt/install.sh
	modified:   ULIP/models/pointnext/PointNeXt/mkdocs.yml
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/.gitignore
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/README.md
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/chamfer_dist/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/chamfer_dist/chamfer.cu
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/chamfer_dist/chamfer_cuda.cpp
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/chamfer_dist/setup.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/chamfer_dist/test.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/emd/.gitignore
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/emd/README.md
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/emd/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/emd/cuda/emd.cpp
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/emd/cuda/emd_kernel.cu
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/emd/emd.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/emd/setup.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/emd/test_emd_loss.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointnet2_batch/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointnet2_batch/setup.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointnet2_batch/src/ball_query.cpp
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointnet2_batch/src/ball_query_gpu.cu
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointnet2_batch/src/ball_query_gpu.h
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointnet2_batch/src/cuda_utils.h
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointnet2_batch/src/group_points.cpp
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointnet2_batch/src/group_points_gpu.cu
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointnet2_batch/src/group_points_gpu.h
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointnet2_batch/src/interpolate.cpp
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointnet2_batch/src/interpolate_gpu.cu
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointnet2_batch/src/interpolate_gpu.h
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointnet2_batch/src/pointnet2_api.cpp
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointnet2_batch/src/sampling.cpp
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointnet2_batch/src/sampling_gpu.cu
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointnet2_batch/src/sampling_gpu.h
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointops/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointops/functions/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointops/functions/pointops.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointops/setup.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointops/src/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointops/src/aggregation/aggregation_cuda.cpp
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointops/src/aggregation/aggregation_cuda_kernel.cu
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointops/src/aggregation/aggregation_cuda_kernel.h
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointops/src/ballquery/ballquery_cuda.cpp
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointops/src/ballquery/ballquery_cuda_kernel.cu
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointops/src/ballquery/ballquery_cuda_kernel.h
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointops/src/cuda_utils.h
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointops/src/grouping/grouping_cuda.cpp
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointops/src/grouping/grouping_cuda_kernel.cu
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointops/src/grouping/grouping_cuda_kernel.h
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointops/src/interpolation/interpolation_cuda.cpp
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointops/src/interpolation/interpolation_cuda_kernel.cu
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointops/src/interpolation/interpolation_cuda_kernel.h
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointops/src/knnquery/knnquery_cuda.cpp
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointops/src/knnquery/knnquery_cuda_kernel.cu
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointops/src/knnquery/knnquery_cuda_kernel.h
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointops/src/pointops_api.cpp
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointops/src/sampling/sampling_cuda.cpp
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointops/src/sampling/sampling_cuda_kernel.cu
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointops/src/sampling/sampling_cuda_kernel.h
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointops/src/subtraction/subtraction_cuda.cpp
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointops/src/subtraction/subtraction_cuda_kernel.cu
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/pointops/src/subtraction/subtraction_cuda_kernel.h
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/subsampling/cpp_utils/cloud/cloud.cpp
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/subsampling/cpp_utils/cloud/cloud.h
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/subsampling/cpp_utils/nanoflann/nanoflann.hpp
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/subsampling/grid_subsampling/grid_subsampling.cpp
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/subsampling/grid_subsampling/grid_subsampling.h
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/subsampling/setup.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/cpp/subsampling/wrapper.cpp
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/atom3d/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/atom3d/psr.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/build.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/data_util.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/datalist.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/dataset_base.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/graph_dataset/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/graph_dataset/graph_dataset.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/graph_dataset/stack_with_pad.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/graph_dataset/structural_dataset.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/graph_dataset/svd_encodings_dataset.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/grid_sample.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/matterport3d/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/matterport3d/category_mapping.tsv
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/matterport3d/matterport3d.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/matterport3d/matterport3d_dataprocessing.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/modelnet/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/modelnet/modelnet40_normal_resampled_loader.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/modelnet/modelnet40_ply_2048_loader.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/molhiv/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/molhiv/data.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/molpcba/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/molpcba/data.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/parsers/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/parsers/class_map.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/parsers/constants.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/parsers/parser.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/parsers/parser_factory.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/parsers/parser_image_folder.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/parsers/parser_image_in_tar.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/parsers/parser_image_tar.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/parsers/parser_tfds.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/pcqm4m/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/pcqm4m/data.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/pcqm4mv2/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/pcqm4mv2/data.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/s3dis/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/s3dis/s3dis.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/s3dis/s3dis_block.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/s3dis/s3dis_sphere.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/scannetv2/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/scannetv2/scannet.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/scanobjectnn/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/scanobjectnn/scanobjectnn.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/semantic_kitti/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/semantic_kitti/compile_op.sh
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/semantic_kitti/helper_tool.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/semantic_kitti/label_mapping.yaml
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/semantic_kitti/preprocess/data_pre.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/semantic_kitti/semantickitti.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/semantic_kitti/utils/6_fold_cv.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/semantic_kitti/utils/cpp_wrappers/compile_wrappers.sh
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/semantic_kitti/utils/cpp_wrappers/cpp_subsampling/grid_subsampling/grid_subsampling.cpp
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/semantic_kitti/utils/cpp_wrappers/cpp_subsampling/grid_subsampling/grid_subsampling.h
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/semantic_kitti/utils/cpp_wrappers/cpp_subsampling/setup.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/semantic_kitti/utils/cpp_wrappers/cpp_subsampling/wrapper.cpp
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/semantic_kitti/utils/cpp_wrappers/cpp_utils/cloud/cloud.cpp
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/semantic_kitti/utils/cpp_wrappers/cpp_utils/cloud/cloud.h
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/semantic_kitti/utils/cpp_wrappers/cpp_utils/nanoflann/nanoflann.hpp
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/semantic_kitti/utils/data_prepare_s3dis.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/semantic_kitti/utils/data_prepare_semantic3d.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/semantic_kitti/utils/data_prepare_semantickitti.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/semantic_kitti/utils/download_semantic3d.sh
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/semantic_kitti/utils/meta/anno_paths.txt
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/semantic_kitti/utils/meta/class_names.txt
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/semantic_kitti/utils/nearest_neighbors/KDTreeTableAdaptor.h
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/semantic_kitti/utils/nearest_neighbors/knn.cpp
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/semantic_kitti/utils/nearest_neighbors/knn.pyx
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/semantic_kitti/utils/nearest_neighbors/knn_.cxx
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/semantic_kitti/utils/nearest_neighbors/knn_.h
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/semantic_kitti/utils/nearest_neighbors/nanoflann.hpp
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/semantic_kitti/utils/nearest_neighbors/setup.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/semantic_kitti/utils/nearest_neighbors/test.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/semantic_kitti/utils/semantic-kitti.yaml
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/shapenet/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/shapenet/shapenet55.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/shapenet/shapenetpart.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/shapenetpart/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/shapenetpart/shapenet55.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/shapenetpart/shapenetpart.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/vis2d.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/dataset/vis3d.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/loss/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/loss/build.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/loss/cross_entropy.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/loss/distill_loss.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/backbone/Stratified_transformer.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/backbone/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/backbone/baafnet.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/backbone/ball_dgcnn.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/backbone/curvenet.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/backbone/deepgcn.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/backbone/dgcnn.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/backbone/graphvit3d.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/backbone/grouppointnet.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/backbone/pct.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/backbone/pointmlp.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/backbone/pointnet.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/backbone/pointnetv2.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/backbone/pointnext.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/backbone/pointnextPyG.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/backbone/pointtransformer.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/backbone/pointvit.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/backbone/randlenet.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/backbone/resnet.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/backbone/simpleview.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/backbone/simpleview_util.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/build.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/classification/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/classification/cls_base.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/classification/point_bert.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/layers/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/layers/activation.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/layers/attention.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/layers/conv.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/layers/drop.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/layers/graph_conv.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/layers/group.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/layers/group_embed.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/layers/helpers.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/layers/kmeans.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/layers/knn.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/layers/local_aggregation.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/layers/mlp.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/layers/norm.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/layers/padding.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/layers/patch_embed.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/layers/registry.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/layers/subsample.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/layers/upsampling.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/layers/weight_init.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/reconstruction/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/reconstruction/base_recontruct.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/reconstruction/maskedpoint.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/reconstruction/maskedpointgroup.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/reconstruction/maskedpointvit.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/reconstruction/nodeshuffle.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/registry.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/segmentation/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/segmentation/base_seg.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/models/segmentation/vit_seg.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/optim/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/optim/adabelief.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/optim/adafactor.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/optim/adahessian.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/optim/adamp.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/optim/adamw.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/optim/lamb.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/optim/lars.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/optim/lookahead.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/optim/madgrad.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/optim/nadam.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/optim/nvnovograd.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/optim/optim_factory.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/optim/radam.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/optim/rmsprop_tf.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/optim/sgdp.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/scheduler/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/scheduler/cosine_lr.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/scheduler/multistep_lr.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/scheduler/plateau_lr.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/scheduler/poly_lr.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/scheduler/scheduler.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/scheduler/scheduler_factory.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/scheduler/step_lr.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/scheduler/tanh_lr.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/transforms/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/transforms/point_transform_cpu.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/transforms/point_transformer_gpu.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/transforms/transforms_factory.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/utils/__init__.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/utils/ckpt_util.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/utils/config.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/utils/dist_utils.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/utils/logger.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/utils/metrics.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/utils/random.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/utils/registry.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/utils/str2bool.py
	modified:   ULIP/models/pointnext/PointNeXt/openpoints/utils/wandb.py
	modified:   ULIP/models/pointnext/PointNeXt/requirements.txt
	modified:   ULIP/models/pointnext/PointNeXt/script/download_s3dis.sh
	modified:   ULIP/models/pointnext/PointNeXt/script/main_classification.sh
	modified:   ULIP/models/pointnext/PointNeXt/script/main_partseg.sh
	modified:   ULIP/models/pointnext/PointNeXt/script/main_segmentation.sh
	modified:   ULIP/models/pointnext/PointNeXt/script/profile_flops.sh
	modified:   ULIP/models/pointnext/PointNeXt/script/test_all_in_one.sh
	modified:   ULIP/models/pointnext/PointNeXt/update.sh
	modified:   ULIP/models/pointnext/pointnext-s.yaml
	modified:   ULIP/models/pointnext/pointnext.py
	modified:   ULIP/scripts/pretrain_pointbert.sh
	modified:   ULIP/scripts/pretrain_pointmlp.sh
	modified:   ULIP/scripts/pretrain_pointnet2_ssg.sh
	modified:   ULIP/scripts/pretrain_pointnext.sh
	modified:   ULIP/scripts/test_pointbert.sh
	modified:   ULIP/scripts/test_pointmlp.sh
	modified:   ULIP/scripts/test_pointnet2_ssg.sh
	modified:   ULIP/scripts/test_pointnext.sh
	modified:   ULIP/scripts/test_ulip2_pointbert_modelnet40.sh
	modified:   ULIP/scripts/test_ulip2_pointbert_objaverse_lvis.sh
	modified:   ULIP/utils/__init__.py
	modified:   ULIP/utils/bpe_simple_vocab_16e6.txt.gz
	modified:   ULIP/utils/build.py
	modified:   ULIP/utils/config.py
	modified:   ULIP/utils/io.py
	modified:   ULIP/utils/logger.py
	modified:   ULIP/utils/registry.py
	modified:   ULIP/utils/tokenizer.py
	modified:   config/README.md
	modified:   config/fusion_config.yaml
	modified:   config/test_config.yaml
	modified:   docs/SETUP_GUIDE_TEMPLATE.md
	modified:   docs/pipeline_usage.md
	modified:   docs/plans/2026-03-27-fusion-pipeline-implementation.md
	modified:   docs/plans/2026-03-27-fusion-pipeline-integration-design.md
	modified:   libs/dataset_loader.py
	modified:   libs/pointnet_extractor.py
	modified:   pointnet_project/Pointnet_Pointnet2_pytorch/data_utils/__init__.py
	modified:   train_utils.py
	modified:   verify_gpu.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	.worktrees/
	README_ABLATION.md
	ablation_results/
	ablation_results_full/
	config/full_train_config.yaml
	docs/FULL_FUSION_PIPELINE_GUIDE.md
	docs/POINTNET2_TRAINING_GUIDE.md
	docs/SETUP_GUIDE.md
	docs/ULIP_TRAINING_GUIDE.md
	docs/backup/
	docs/plans/2026-03-28-fusion-pipeline-improvements.md
	extraction.log
	libs/__init__.py
	pipeline.pid
	real_ulip_eval.log
	rename_imports.py
	run_ablation.sh
	test_ablation/
	test_circular.py
	test_circular2.py
	test_import.py
	test_run_full/
	test_ulip_load.py
	test_ulip_real.py
	test_utils_import.py
	tests/verification/verify_end_to_end.py
	tests/verification/verify_full_extraction.py
	tests/verification/verify_fusion_training.py
	tests/verification/verify_pointnet_extraction.py
	tests/verification/verify_ulip_extraction.py

no changes added to commit (use "git add" and/or "git commit -a")
```

## Directory Structure
```
.
./ablation_features
./ablation_results
./ablation_results_full
./ablation_results_full/fusion
./ablation_results_full/fusion/checkpoints
./ablation_results_full/pointnet_only
./ablation_results_full/pointnet_only/checkpoints
./ablation_results_full/ulip_only
./ablation_results_full/ulip_only/checkpoints
./ablation_results/fusion
./ablation_results/fusion/checkpoints
./ablation_results/pointnet_only
./ablation_results/pointnet_only/checkpoints
./ablation_results/ulip_only
./ablation_results/ulip_only/checkpoints
./checkpoints
./config
./docs
./docs/backup
./docs/plans
./feature_cache
./full_training_checkpoints
./full_training_output
./fusion_experiments
./fusion_experiments/fusion_full_50epochs
./fusion_experiments/fusion_full_50epochs/checkpoints
./fusion_experiments/fusion_full_50epochs/training_output
./fusion_experiments/fusion_pipeline_20260407_174152
./fusion_experiments/fusion_pipeline_20260407_174152/checkpoints
./fusion_experiments/fusion_pipeline_20260407_174152/features
./fusion_experiments/fusion_pipeline_20260407_174152/training_output
./.git
./.git/branches
./.git/hooks
./.git/info
./.git/logs
./.git/logs/refs
./.git/objects
./.git/objects/00
./.git/objects/01
./.git/objects/03
./.git/objects/04
./.git/objects/05
./.git/objects/06
./.git/objects/07
./.git/objects/08
./.git/objects/09
./.git/objects/0a
./.git/objects/0b
./.git/objects/0c
./.git/objects/0d
./.git/objects/0e
./.git/objects/0f
./.git/objects/10
./.git/objects/11
./.git/objects/12
./.git/objects/15
./.git/objects/16
./.git/objects/17
./.git/objects/18
./.git/objects/19
./.git/objects/1a
./.git/objects/1b
./.git/objects/1c
./.git/objects/1d
./.git/objects/1e
./.git/objects/1f
./.git/objects/20
./.git/objects/21
./.git/objects/22
./.git/objects/23
./.git/objects/24
./.git/objects/25
./.git/objects/27
./.git/objects/28
./.git/objects/29
./.git/objects/2a
./.git/objects/2b
./.git/objects/2c
./.git/objects/2d
./.git/objects/2e
./.git/objects/2f
./.git/objects/30
./.git/objects/31
./.git/objects/32
./.git/objects/34
./.git/objects/35
./.git/objects/37
./.git/objects/38
./.git/objects/39
./.git/objects/3a
./.git/objects/3b
./.git/objects/3c
./.git/objects/3d
./.git/objects/3f
./.git/objects/40
./.git/objects/41
./.git/objects/42
./.git/objects/43
./.git/objects/44
./.git/objects/45
./.git/objects/46
./.git/objects/47
./.git/objects/48
./.git/objects/49
./.git/objects/4b
./.git/objects/4d
./.git/objects/4e
./.git/objects/4f
./.git/objects/50
./.git/objects/51
./.git/objects/52
./.git/objects/53
./.git/objects/55
./.git/objects/56
./.git/objects/57
./.git/objects/58
./.git/objects/59
./.git/objects/5a
./.git/objects/5b
./.git/objects/5c
./.git/objects/5d
./.git/objects/5f
./.git/objects/60
./.git/objects/61
./.git/objects/62
./.git/objects/63
./.git/objects/64
./.git/objects/65
./.git/objects/66
./.git/objects/67
./.git/objects/68
./.git/objects/69
./.git/objects/6a
./.git/objects/6b
./.git/objects/6c
./.git/objects/6d
./.git/objects/6e
./.git/objects/6f
./.git/objects/70
./.git/objects/71
./.git/objects/72
./.git/objects/73
./.git/objects/74
./.git/objects/76
./.git/objects/77
./.git/objects/78
./.git/objects/79
./.git/objects/7b
./.git/objects/7c
./.git/objects/7d
./.git/objects/7e
./.git/objects/7f
./.git/objects/80
./.git/objects/81
./.git/objects/82
./.git/objects/83
./.git/objects/84
./.git/objects/85
./.git/objects/86
./.git/objects/87
./.git/objects/88
./.git/objects/89
./.git/objects/8a
./.git/objects/8b
./.git/objects/8c
./.git/objects/8d
./.git/objects/8e
./.git/objects/8f
./.git/objects/90
./.git/objects/91
./.git/objects/92
./.git/objects/93
./.git/objects/94
./.git/objects/95
./.git/objects/96
./.git/objects/97
./.git/objects/98
./.git/objects/99
./.git/objects/9a
./.git/objects/9b
./.git/objects/9d
./.git/objects/9e
./.git/objects/9f
./.git/objects/a0
./.git/objects/a1
./.git/objects/a2
./.git/objects/a3
./.git/objects/a4
./.git/objects/a6
./.git/objects/a7
./.git/objects/a8
./.git/objects/a9
./.git/objects/aa
./.git/objects/ab
./.git/objects/ac
./.git/objects/ad
./.git/objects/ae
./.git/objects/af
./.git/objects/b0
./.git/objects/b1
./.git/objects/b2
./.git/objects/b3
./.git/objects/b4
./.git/objects/b5
./.git/objects/b6
./.git/objects/b7
./.git/objects/b8
./.git/objects/b9
./.git/objects/ba
./.git/objects/bb
./.git/objects/bc
./.git/objects/bd
./.git/objects/be
./.git/objects/bf
./.git/objects/c0
./.git/objects/c2
./.git/objects/c3
./.git/objects/c4
./.git/objects/c5
./.git/objects/c6
./.git/objects/c7
./.git/objects/c8
./.git/objects/c9
./.git/objects/ca
./.git/objects/cb
./.git/objects/cc
./.git/objects/cd
./.git/objects/ce
./.git/objects/cf
./.git/objects/d0
./.git/objects/d1
./.git/objects/d2
./.git/objects/d3
./.git/objects/d4
./.git/objects/d5
./.git/objects/d6
./.git/objects/d7
./.git/objects/d8
./.git/objects/d9
./.git/objects/da
./.git/objects/db
./.git/objects/dc
./.git/objects/dd
./.git/objects/de
./.git/objects/df
./.git/objects/e0
./.git/objects/e1
./.git/objects/e2
./.git/objects/e3
./.git/objects/e5
./.git/objects/e6
./.git/objects/e7
./.git/objects/e8
./.git/objects/e9
./.git/objects/ea
./.git/objects/eb
./.git/objects/ec
./.git/objects/ed
./.git/objects/ee
./.git/objects/ef
./.git/objects/f0
./.git/objects/f1
./.git/objects/f2
./.git/objects/f3
./.git/objects/f5
./.git/objects/f6
./.git/objects/f7
./.git/objects/f8
./.git/objects/f9
./.git/objects/fa
./.git/objects/fb
./.git/objects/fc
./.git/objects/fd
./.git/objects/fe
./.git/objects/ff
./.git/objects/info
./.git/objects/pack
./.git/refs
./.git/refs/heads
./.git/refs/tags
./.git/worktrees
./.git/worktrees/improve-fusion
./libs
./libs/__pycache__
./pointnet_project
./pointnet_project/Pointnet_Pointnet2_pytorch
./pointnet_project/Pointnet_Pointnet2_pytorch/data
./pointnet_project/Pointnet_Pointnet2_pytorch/data_utils
./pointnet_project/Pointnet_Pointnet2_pytorch/log
./pointnet_project/Pointnet_Pointnet2_pytorch/models
./pointnet_project/Pointnet_Pointnet2_pytorch/visualizer
./project_utils
./project_utils/__pycache__
./__pycache__
./.pytest_cache
./.pytest_cache/v
./.pytest_cache/v/cache
./scripts
./scripts/__pycache__
./test_ablation
./test_ablation/fusion
./test_ablation/fusion/checkpoints
./test_ablation/pointnet_only
./test_ablation/pointnet_only/checkpoints
./test_ablation/ulip_only
./test_ablation/ulip_only/checkpoints
./test_run_full
./test_run_full/checkpoints
./test_run_full/features
./test_run_full/training_output
./tests
./tests/__pycache__
./tests/verification
./tests/verification/__pycache__
./training_output
./ULIP
./ULIP/assets
./ULIP/data
./ULIP/data/__pycache__
./ULIP/models
./ULIP/models/customized_backbone
./ULIP/models/pointbert
./ULIP/models/pointmlp
./ULIP/models/pointnet2
./ULIP/models/pointnext
./ULIP/models/__pycache__
./ULIP/scripts
./ULIP/utils
./ULIP/utils/__pycache__
./.worktrees
./.worktrees/improve-fusion
./.worktrees/improve-fusion/checkpoints
./.worktrees/improve-fusion/config
./.worktrees/improve-fusion/docs
./.worktrees/improve-fusion/libs
./.worktrees/improve-fusion/pointnet_project
./.worktrees/improve-fusion/.pytest_cache
./.worktrees/improve-fusion/scripts
./.worktrees/improve-fusion/test_run_full
./.worktrees/improve-fusion/tests
./.worktrees/improve-fusion/training_output
./.worktrees/improve-fusion/ULIP
./.worktrees/improve-fusion/utils
```

## Key Files to Preserve
1. Core code: scripts/, libs/, project_utils/, tests/, fusion_model.py, train_utils.py
2. Real ULIP-2 experiment: fusion_experiments/fusion_pipeline_20260407_174152/
3. Configuration: config/fusion_config.yaml
4. Checkpoints: checkpoints/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt
5. Documentation: docs/ directory

## Files to Remove
1. Temporary test files: test_*.py
2. Duplicate ablation directories: test_ablation/, ablation_results_full/
3. Debug directories: test_run_full/
4. Worktrees: .worktrees/ (except potentially needed for future work)