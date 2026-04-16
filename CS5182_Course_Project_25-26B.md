CS5182 Course Project
(2025/2026 Semester B)
1 Objectives
The objective of this project is to provide students with hands-on experience in recent com-
puter graphics research, specifically focusing on deep learning-based methods. Key areas of
exploration include deep modeling of 3D geometric data, neural rendering, and 2D/3D content
generation.
2 Requirements
Students are invited to explore any learning-based topic within the field of computer graphics,
encompassing both algorithm implementation and performance analysis. To facilitate efficient
training and evaluation, we strongly recommend using a machine equipped with a GPU. Both
PyTorch and TensorFlow frameworks are permitted. A list of reference topics is provided below.
### List of 3D Geometric Data Processing :
• Point cloud shape classification/part segmentation/scene segmentation/...
• Point cloud upsampling/downsampling/registration/compression/denoising/completion/...
• Object detection in 3D point clouds
• 3D surface reconstruction
• ...
### List of Neural Rendering topics:
• Faster Inference
• Faster Training
• Unconstrained Images
• Deformable
• Generalization
• Lighting
• 3D Reconstruction
• ...
3D Content Generation from Images/Texts/...
We also encourage students to propose alternative tasks. A list of references is provided
at the end of this document, and students are welcome to utilize open-source implementations
from GitHub.
This project may be completed individually or in groups of up to three members. Please
note that group projects are expected to demonstrate a higher workload and greater creativity.
1

To ensure accountability, the final report must clearly outline the specific responsibilities and
contributions of each member.
The project requirements are divided into two levels—basic and advanced—to accommodate
students with varying backgrounds. The Basic Requirements (50%) provide an opportunity for
all students to practice deep learning methods in computer graphics. The Advanced Require-
ments (20%) are designed for students who wish to explore and propose novel algorithms. The
remaining 30% is allocated to the presentation component.
2.1 Basic Requirements (50%)
1. Introduction of the project
Students are required to provide an introduction to the selected task, including a con-
cise definition of the input and output requirements. This section must also specify the
dataset utilized, the reference paper for the algorithm, the project’s objective and moti-
vation, and the technical challenges addressed.
2. Deploy deep learning environment
Students should install deep learning platforms, such as TensorFlow or PyTorch, on their
own computers or remote servers. One can refer to Anaconda for environmental manage-
ment.
3. Run Demo code
You will need to consult research papers to select an appropriate algorithm for your task.
Many of these papers offer demo code on GitHub, often including training scripts or
evaluation code with pre-trained models. Students are required to successfully run this
demo code and provide a detailed description of the purpose and function of each step.
4. Train model
Students are expected to download the necessary dataset and train their own models.
Please note that this process may take several hours, even when using a GPU. We strongly
advise against training on a CPU, as this will significantly extend the training duration.
5. Compare your experimental results with those provided by the corresponding
paper
Students are responsible for evaluating the performance of both the trained model and,
if provided, the pre-trained model. It is important to analyze the quantitative error and
accuracy of the models. Additionally, students are expected to showcase the visualized
results. For viewing 3D point clouds, MeshLab can be referred to as a suitable tool.
6. Analyze the performance on other dataset
Students should pick another dataset to verify the performance and generality of this
method.
7. Analyze the drawbacks of the method
Students should carefully examine the failure cases of the algorithm and analyze the
reasons. Some ideas should be proposed to solve the problem.
8. Implement and compare with other state-of-the-art methods
For the same task, students should pick another deep learning-based algorithm, implement
it, and compare the results.
2

2.2 Advanced Requirements (20%)
Students are expected to extend the algorithm by applying it to new tasks or enhancing it
through a proposed method. This may involve modifying the network architecture, loss func-
tion, or dataset. It is essential to clearly explain the rationale behind these modifications and
report the final results.
2.3 Presentation (30%)
The project should be presented during the lecture/tutorial time. Each project has 7 minutes
(7 m presentation + 3 m Q&A). Tentative Presentation Date: 17 and 24 April 2026
3 Submission Details
Each group needs to submit the following items via Canvas, and the submission link in Canvas
will be open later. The submission deadline is 26 April 2026 .
### Program:
1. A source subdirectory containing all the source files, do not include the platform source
code and training dataset.
2. Txt files which contain the results for training and evaluation periods.
3. Data that are evaluated in your report.
### Report:
1. A cover that indicates your name(s) and student ID(s).
2. Describe the results listed in requirements in last section.
4 Marking
This course project contributes 25% of the final course mark.
5 References
### • A comprehensive collection of papers on deep learning-based 3D point cloud processing:
https://github.com/NUAAXQ/awesome-point-cloud-analysis-2023
### • Neural Rendering Courses and Survey:
https://github.com/weihaox/awesome-neural-rendering/blob/master/docs/INTRODUCTION-AND-SURVEY.
md
• 3D Content generation. https://github.com/topics/3d-generation
• Libraries for Geometry Processing
https://github.com/zishun/awesome-geometry-processing
• Advanced computer vision papers: Arxiv https://arxiv.org/list/cs.CV/recent
• Open source code: Github http://github.com/
3

• Environment management: Anaconda https://www.anaconda.com/
• TensorFlow API https://www.tensorflow.org/api_docs/python/tf/
• PyTorch API https://pytorch.org/docs/stable/index.html
• 3D point cloud viewer: MeshLab http://www.meshlab.net/
4