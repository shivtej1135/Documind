from vision.vision_answer_generator import VisionAnswerGenerator

generator = VisionAnswerGenerator()

result = generator.answer_from_image(
    question="What does Figure 6 compare on CIFAR-10?",
    image_path="data/images/Denseresultstable_p8_fig7.png",
    caption="Figure 6. Training on CIFAR-10."
)

print(result)