cmake_minimum_required(VERSION 3.22)
project(CHORUSPlugin VERSION 0.1.0)

add_subdirectory(${JUCE_DIR} JUCE)

juce_add_plugin(CHORUSPlugin
    COMPANY_NAME "CHORUS"
    PLUGIN_MANUFACTURER_CODE Chrs
    PLUGIN_CODE Chr1
    FORMATS VST3
    PRODUCT_NAME "CHORUS")

target_sources(CHORUSPlugin PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp)

set(CHORUS_RUST_PROFILE $<IF:$<CONFIG:Debug>,debug,release>)
set(CHORUS_FFI_LIB ${CMAKE_CURRENT_SOURCE_DIR}/../rust/chorus-ffi/target/${CHORUS_RUST_PROFILE}/libchorus_ffi.a)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/../rust/chorus-ffi/target/release/libchorus_ffi.a
    COMMAND cargo build --manifest-path ${CMAKE_CURRENT_SOURCE_DIR}/../rust/chorus-ffi/Cargo.toml --release
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)
add_custom_target(CHORUSRustFFI DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../rust/chorus-ffi/target/release/libchorus_ffi.a)
add_dependencies(CHORUSPlugin CHORUSRustFFI)

target_compile_features(CHORUSPlugin PRIVATE cxx_std_17)
target_link_libraries(CHORUSPlugin PRIVATE juce::juce_audio_utils ${CHORUS_FFI_LIB})

add_executable(CHORUSPluginSmoke tests/plugin_smoke.cpp)
target_link_libraries(CHORUSPluginSmoke PRIVATE CHORUSPlugin juce::juce_audio_utils)
add_dependencies(CHORUSPluginSmoke CHORUSRustFFI)
enable_testing()
add_test(NAME CHORUSPluginSmoke COMMAND CHORUSPluginSmoke)
