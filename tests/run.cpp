// test_encoder.cpp

#include "gtest/gtest.h"
#include "../lib/encoder.h"

const char* valid_model_file_path = "../ggml-model.bin";
const char* invalid_model_file_path = "invalid-path";

TEST(EncoderTest, InitFromFileNoState_ValidModelFile) {
    struct encoder_context * ctx = encoder_init_from_file_no_state(valid_model_file_path);
    EXPECT_NE(ctx, nullptr);  // If model loading is successful, the context should not be null.
    if (ctx) {
        encoder_context_destroy(ctx);
    }
}

TEST(EncoderTest, InitFromFileNoState_InvalidModelFile) {
    struct encoder_context * ctx = encoder_init_from_file_no_state(invalid_model_file_path);
    EXPECT_EQ(ctx, nullptr);  // If model loading fails, the context should be null.
}

// TODO: Add more tests as needed

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
