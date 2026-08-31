#include "serve/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <string>

namespace {

using Json = nlohmann::json;

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int check(bool condition, const std::string& message) { return condition ? 0 : fail(message); }

int test_single_call() {
    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output("Calling weather.\n"
                                                   "<tool_call>\n"
                                                   "<function=get_weather>\n"
                                                   "<parameter=city>\nParis\n</parameter>\n"
                                                   "<parameter=days>\n2\n</parameter>\n"
                                                   "</function>\n"
                                                   "</tool_call>",
                                                   64);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "single call parsed as tool response");
    failures += check(parsed.content == "Calling weather.", "content prefix trimmed");
    failures += check(parsed.tool_calls.size() == 1, "one parsed call");
    failures += check(parsed.tool_calls[0].id.rfind("call_", 0) == 0, "generated call id prefix");
    failures += check(parsed.tool_calls[0].name == "get_weather", "function name parsed");
    const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(args.at("city") == "Paris", "string parameter parsed");
    failures += check(args.at("days") == 2, "number parameter parsed");
    return failures;
}

int test_multiple_calls_and_json_values() {
    const ninfer::serve::ParsedToolCallOutput parsed = ninfer::serve::parse_qwen_tool_call_output(
        "<tool_call>\n"
        "<function=first>\n"
        "<parameter=payload>\n{\"ok\":true,\"items\":[1,2]}\n</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "<tool_call>\n"
        "<function=second>\n"
        "<parameter=value>\nplain text\n</parameter>\n"
        "</function>\n"
        "</tool_call>",
        64);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "multiple calls parsed as tool response");
    failures += check(parsed.tool_calls.size() == 2, "two parsed calls");
    failures += check(parsed.tool_calls[0].name == "first", "first call name");
    failures += check(parsed.tool_calls[1].name == "second", "second call name");
    const Json first = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(first.at("payload").at("ok") == true, "object parameter bool");
    failures += check(first.at("payload").at("items").at(1) == 2, "object parameter array");
    const Json second = Json::parse(parsed.tool_calls[1].arguments_json);
    failures += check(second.at("value") == "plain text", "plain text parameter string");
    return failures;
}

int test_malformed_falls_back_to_text() {
    const std::string text = "<tool_call>\n<function=get_weather>\n";
    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output(text, 64);
    int failures = 0;
    failures += check(!parsed.is_tool_call_response, "malformed xml is not tool response");
    failures += check(parsed.content.find("<tool_call>") == std::string::npos,
                      "malformed xml leaked into content");
    failures += check(parsed.tool_calls.empty(), "malformed xml has no calls");
    return failures;
}

int test_suffix_after_tool_becomes_content() {
    // A well-formed call followed by non-whitespace prose: the call is salvaged and the suffix
    // becomes content. It is never a fallback to raw text -- the parser's contract is that once a
    // <tool_call> marker is confirmed, its XML is never echoed to the client.
    const std::string text = "<tool_call>\n"
                             "<function=get_weather>\n"
                             "<parameter=city>\nParis\n</parameter>\n"
                             "</function>\n"
                             "</tool_call>\n"
                             "extra answer";
    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output(text, 64);
    int failures = 0;
    failures += check(parsed.is_tool_call_response, "suffix after a valid call keeps tool response");
    failures += check(parsed.tool_calls.size() == 1, "suffix after a valid call keeps one call");
    failures += check(parsed.tool_calls.size() == 1 && parsed.tool_calls[0].name == "get_weather",
                      "salvaged call name with a suffix");
    failures += check(parsed.content == "extra answer", "suffix kept verbatim as content");
    failures += check(parsed.content.find("<tool_call>") == std::string::npos,
                      "tool-call xml leaked into content alongside the suffix");
    return failures;
}

int test_configured_name_limit() {
    const std::string name(128, 'a');
    const std::string text = "<tool_call>\n<function=" + name + ">\n</function>\n</tool_call>";

    const ninfer::serve::ParsedToolCallOutput anthropic =
        ninfer::serve::parse_qwen_tool_call_output(text, 128);
    const ninfer::serve::ParsedToolCallOutput openai =
        ninfer::serve::parse_qwen_tool_call_output(text, 64);
    const std::string too_long_text =
        "<tool_call>\n<function=" + std::string(129, 'a') + ">\n</function>\n</tool_call>";
    const ninfer::serve::ParsedToolCallOutput too_long =
        ninfer::serve::parse_qwen_tool_call_output(too_long_text, 128);

    int failures = 0;
    failures += check(anthropic.is_tool_call_response && anthropic.tool_calls.size() == 1 &&
                          anthropic.tool_calls[0].name == name,
                      "128-character name accepted with Anthropic limit");
    failures +=
        check(!openai.is_tool_call_response, "128-character name rejected with OpenAI limit");
    failures +=
        check(!too_long.is_tool_call_response, "129-character name rejected with Anthropic limit");
    return failures;
}

int test_incremental_filter_valid_tool() {
    ninfer::serve::ToolCallStreamFilter filter;
    std::string visible;
    visible += filter.feed("Calling weather.  \n<tool_");
    visible += filter.feed("call>\n<function=get_weather>");
    visible += filter.feed("\n</function>\n</tool_call>");
    visible += filter.finish(true);
    int failures = 0;
    failures += check(visible == "Calling weather.",
                      "valid tool filter did not stream the trimmed content prefix");
    failures +=
        check(filter.emitted_bytes() == visible.size(), "valid tool filter byte count mismatch");
    return failures;
}

int test_incremental_filter_fallback() {
    // Once a full <tool_call> marker is recognized, the buffered region belongs to the
    // tool-call protocol and must never be replayed verbatim, even if the call turns out
    // malformed/unclosed: only the prefix seen before the marker stays visible.
    const std::string original = "prefix  \n<tool_call>\n<function=broken>";
    ninfer::serve::ToolCallStreamFilter malformed;
    std::string restored;
    restored += malformed.feed(original.substr(0, 10));
    restored += malformed.feed(original.substr(10));
    restored += malformed.finish(false);

    ninfer::serve::ToolCallStreamFilter normal;
    std::string ordinary;
    ordinary += normal.feed("ordinary text  ");
    ordinary += normal.finish(false);

    int failures = 0;
    failures += check(restored == "prefix", "malformed tool call XML leaked after full marker");
    failures +=
        check(ordinary == "ordinary text  ", "ordinary filtered output lost trailing whitespace");
    return failures;
}

int test_incremental_filter_partial_marker_stays_visible() {
    // A trailing fragment that never completes the "<tool_call>" marker is ordinary text,
    // not a protocol region, and must be replayed verbatim on finish(false).
    ninfer::serve::ToolCallStreamFilter filter;
    std::string visible;
    visible += filter.feed("Hello <tool_");
    visible += filter.finish(false);

    int failures = 0;
    failures += check(visible == "Hello <tool_",
                      "incomplete <tool_call> marker suffix was lost or altered");
    failures +=
        check(filter.emitted_bytes() == visible.size(), "partial marker byte count mismatch");
    return failures;
}

int test_incremental_filter_matches_parser_on_malformed_marker() {
    // Regression for the streamed_content_bytes > outcome.text.size() invariant crash:
    // the marker is split across feed() calls, the block is unclosed/malformed, and no
    // call is salvaged. The filter's visible output must equal the final parser's content
    // byte-for-byte, not just be a subset, so streamed_content_bytes can never outgrow
    // outcome.text.size().
    const std::string raw = "Hello world <tool_call>\n<function=broken>";

    ninfer::serve::ToolCallStreamFilter filter;
    std::string streamed_visible;
    streamed_visible += filter.feed(raw.substr(0, 15));
    streamed_visible += filter.feed(raw.substr(15));

    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output(raw, 64);
    streamed_visible += filter.finish(parsed.is_tool_call_response);

    int failures = 0;
    failures += check(!parsed.is_tool_call_response, "malformed/unclosed block salvaged unexpectedly");
    failures += check(streamed_visible.find("<tool_call>") == std::string::npos,
                      "malformed tool-call XML emitted as visible content");
    failures += check(filter.emitted_bytes() == streamed_visible.size(),
                      "emitted_bytes() does not match actual visible bytes");
    failures += check(streamed_visible == parsed.content,
                      "streamed visible text diverged from final parsed.content");
    failures += check(filter.emitted_bytes() <= parsed.content.size(),
                      "streamed_content_bytes would exceed outcome.text.size()");
    return failures;
}


int test_trailing_prose_keeps_calls() {
    // Model narrates after its tool call ("response", analysis text). The well-formed call
    // must still be salvaged; trailing prose becomes content instead of poisoning the parse.
    const std::string text = "<tool_call>\n"
                             "<function=bash>\n"
                             "<parameter=command>\nls\n</parameter>\n"
                             "</function>\n"
                             "</tool_call>\n"
                             "The exit code is 2 because src does not exist.";
    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output(text, 64);
    int failures = 0;
    failures += check(parsed.is_tool_call_response, "trailing prose keeps tool response");
    failures += check(parsed.tool_calls.size() == 1, "trailing prose keeps one call");
    failures += check(parsed.tool_calls[0].name == "bash", "salvaged call name");
    failures += check(parsed.content.find("exit code is 2") != std::string::npos,
                      "trailing prose preserved as content");
    return failures;
}

int test_prose_between_blocks_salvages_first() {
    // First block good, then prose, then a second marker that is never closed: first call
    // survives, prose and fragment become content.
    const std::string text = "<tool_call>\n"
                             "<function=a>\n"
                             "<parameter=x>1</parameter>\n"
                             "</function>\n"
                             "</tool_call>\n"
                             "thinking out loud <tool_call>\n<function=b>";
    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output(text, 64);
    int failures = 0;
    failures += check(parsed.is_tool_call_response, "prose between blocks keeps response");
    failures += check(parsed.tool_calls.size() == 1, "only first block salvaged");
    failures += check(parsed.tool_calls[0].name == "a", "first block name");
    failures += check(parsed.content.find("thinking out loud") != std::string::npos &&
                          parsed.content.find("<tool_call>") != std::string::npos,
                      "prose and unclosed fragment kept as content");
    return failures;
}

int test_all_blocks_bad_drops_xml() {
    // A single un-closable block is dropped; its raw XML must never leak into content.
    const std::string text = "prefix\n<tool_call>\n<function=broken>\n<parameter=x>1";
    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output(text, 64);
    int failures = 0;
    failures += check(!parsed.is_tool_call_response, "all-bad blocks are not tool calls");
    failures += check(parsed.tool_calls.empty(), "all-bad blocks yield no calls");
    failures += check(parsed.content.find("<tool_call>") == std::string::npos,
                      "malformed block XML leaked into content");
    failures += check(parsed.content.find("prefix") != std::string::npos,
                      "prose before blocks was lost");
    return failures;
}

int test_malformed_block_dropped_good_neighbors_survive() {
    const std::string good =
        "<tool_call>\n<function=bash>\n<parameter=cmd>\nls\n</parameter>\n</function>\n</tool_call>";
    const std::string bad_tail =
        "<tool_call>\n<function=bash>\n<parameter=cmd>\nls\n</parameter>\n";
    const std::string text = good + "\n" + bad_tail + "\nsome prose";
    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output(text, 64);
    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "good block before malformed one was not salvaged");
    failures += check(parsed.content.find("<tool_call>") == std::string::npos &&
                          parsed.content.find("</function>") == std::string::npos,
                      "malformed block XML leaked into content");
    return failures;
}

int test_malformed_param_block_dropped() {
    const std::string text =
        "note\n<tool_call>\n<function=bash>\n<parameter cmd>\nls\n</parameter>"
        "\n</function>\n</tool_call>";
    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output(text, 64);
    int failures = 0;
    failures += check(!parsed.is_tool_call_response && parsed.tool_calls.empty(),
                      "single malformed-param block should not parse");
    failures += check(parsed.content.find("<tool_call>") == std::string::npos,
                      "malformed-param block XML leaked into content");
    failures += check(parsed.content.find("note") != std::string::npos,
                      "prose before the block was lost");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_single_call();
    failures += test_multiple_calls_and_json_values();
    failures += test_malformed_falls_back_to_text();
    failures += test_suffix_after_tool_becomes_content();
    failures += test_configured_name_limit();
    failures += test_incremental_filter_valid_tool();
    failures += test_incremental_filter_fallback();
    failures += test_incremental_filter_partial_marker_stays_visible();
    failures += test_incremental_filter_matches_parser_on_malformed_marker();
    failures += test_malformed_block_dropped_good_neighbors_survive();
    failures += test_malformed_param_block_dropped();
    failures += test_trailing_prose_keeps_calls();
    failures += test_prose_between_blocks_salvages_first();
    failures += test_all_blocks_bad_drops_xml();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
