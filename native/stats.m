// stats: in ra 1 dòng JSON mỗi ~500ms với CPU %, GPU %, RAM, công suất CPU/GPU/ANE và nhiệt độ.
// Không cần sudo: IOReport (công suất CPU/GPU/ANE), SMC (công suất SoC, toàn máy, nhiệt độ), IOAccelerator (GPU %).
#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
#include <mach/mach.h>
#include <sys/sysctl.h>

// ---- IOReport (private, /usr/lib/libIOReport.dylib) ----
typedef struct IOReportSubscriptionRef *IOReportSubscriptionRef;
extern CFMutableDictionaryRef IOReportCopyChannelsInGroup(CFStringRef, CFStringRef, uint64_t, uint64_t, uint64_t);
extern IOReportSubscriptionRef IOReportCreateSubscription(void *, CFMutableDictionaryRef, CFMutableDictionaryRef *, uint64_t, CFTypeRef);
extern CFDictionaryRef IOReportCreateSamples(IOReportSubscriptionRef, CFMutableDictionaryRef, CFTypeRef);
extern CFDictionaryRef IOReportCreateSamplesDelta(CFDictionaryRef, CFDictionaryRef, CFTypeRef);
extern CFStringRef IOReportChannelGetGroup(CFDictionaryRef);
extern CFStringRef IOReportChannelGetChannelName(CFDictionaryRef);
extern CFStringRef IOReportChannelGetUnitLabel(CFDictionaryRef);
extern int64_t IOReportSimpleGetIntegerValue(CFDictionaryRef, int32_t);

// ---- SMC (AppleSMC user client): công suất CPU/toàn máy + nhiệt độ ----
typedef struct { char major, minor, build, reserved[1]; uint16_t release; } SMCVers;
typedef struct { uint16_t version, length; uint32_t cpuPLimit, gpuPLimit, memPLimit; } SMCPLimit;
typedef struct { uint32_t dataSize, dataType; uint8_t dataAttributes; } SMCKeyInfo;
typedef struct { uint32_t key; SMCVers vers; SMCPLimit pLimitData; SMCKeyInfo keyInfo; uint8_t result, status, data8; uint32_t data32; uint8_t bytes[32]; } SMCParam;
static io_connect_t smcConn = 0;
static int smcCall(SMCParam *in, SMCParam *out) { size_t sz = sizeof(SMCParam); return IOConnectCallStructMethod(smcConn, 2, in, sizeof(SMCParam), out, &sz); }
static uint32_t fourcc(const char *s) { return ((uint32_t)s[0] << 24) | ((uint32_t)s[1] << 16) | ((uint32_t)s[2] << 8) | (uint32_t)s[3]; }
static int smcKeyInfo(uint32_t key, SMCKeyInfo *info) {
    SMCParam in = {0}, out = {0}; in.key = key; in.data8 = 9;
    if (smcCall(&in, &out) || out.result) return -1;
    *info = out.keyInfo; return 0;
}
static float smcReadFloat(uint32_t key, uint32_t size) {
    SMCParam in = {0}, out = {0}; in.key = key; in.keyInfo.dataSize = size; in.data8 = 5;
    if (smcCall(&in, &out) || out.result) return -1;
    float f; memcpy(&f, out.bytes, 4); return f;
}
#define MAX_KEYS 1024
static uint32_t cpuTempKeys[MAX_KEYS], gpuTempKeys[MAX_KEYS]; static int nCpuTemp = 0, nGpuTemp = 0;
static uint32_t socPowerKey = 0, sysPowerKey = 0;
static BOOL haveCpu = NO;  // đã từng thấy năng lượng CPU > 0 từ IOReport

static void smcInit(void) {
    io_service_t svc = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AppleSMC"));
    if (!svc || IOServiceOpen(svc, mach_task_self(), 0, &smcConn)) { smcConn = 0; return; }
    SMCKeyInfo info; if (smcKeyInfo(fourcc("#KEY"), &info)) return;
    SMCParam in = {0}, out = {0}; in.key = fourcc("#KEY"); in.keyInfo.dataSize = info.dataSize; in.data8 = 5;
    if (smcCall(&in, &out)) return;
    uint32_t n = ((uint32_t)out.bytes[0] << 24) | ((uint32_t)out.bytes[1] << 16) | ((uint32_t)out.bytes[2] << 8) | out.bytes[3];
    for (uint32_t i = 0; i < n; i++) {
        SMCParam a = {0}, b = {0}; a.data8 = 8; a.data32 = i;
        if (smcCall(&a, &b)) continue;
        uint32_t key = b.key; char c0 = key >> 24, c1 = (key >> 16) & 0xff;
        if (c0 != 'T' && c0 != 'P') continue;
        if (smcKeyInfo(key, &info) || info.dataType != fourcc("flt ") || info.dataSize != 4) continue;
        if (c0 == 'T' && c1 == 'p' && nCpuTemp < MAX_KEYS) cpuTempKeys[nCpuTemp++] = key;
        else if (c0 == 'T' && c1 == 'g' && nGpuTemp < MAX_KEYS) gpuTempKeys[nGpuTemp++] = key;
        else if (key == fourcc("PSVR")) socPowerKey = key;  // công suất SoC (CPU + GPU + ANE + fabric)
        else if (key == fourcc("PD0R") || (key == fourcc("PSTR") && !sysPowerKey)) sysPowerKey = key;
    }
}
static double smcAvg(uint32_t *keys, int n) {
    double sum = 0; int cnt = 0;
    for (int i = 0; i < n; i++) { float v = smcReadFloat(keys[i], 4); if (v > 5 && v < 130) { sum += v; cnt++; } }
    return cnt ? sum / cnt : -1;
}

static double energyToJoules(int64_t v, NSString *unit) {
    if ([unit isEqualToString:@"nJ"]) return v / 1e9;
    if ([unit isEqualToString:@"uJ"]) return v / 1e6;
    if ([unit isEqualToString:@"mJ"]) return v / 1e3;
    return v / 1e3;
}

static double gpuUtilization(void) {
    double util = -1;
    io_iterator_t it;
    if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOAccelerator"), &it) != KERN_SUCCESS) return -1;
    io_object_t obj;
    while ((obj = IOIteratorNext(it))) {
        CFTypeRef stats = IORegistryEntryCreateCFProperty(obj, CFSTR("PerformanceStatistics"), kCFAllocatorDefault, 0);
        if (stats) {
            NSDictionary *d = (__bridge_transfer NSDictionary *)stats;
            NSNumber *n = d[@"Device Utilization %"];
            if (n) util = MAX(util, n.doubleValue);
        }
        IOObjectRelease(obj);
    }
    IOObjectRelease(it);
    return util;
}

static processor_info_array_t prevCpu = NULL;
static mach_msg_type_number_t prevCpuCount = 0;

static double cpuUsage(void) {
    natural_t ncpu; processor_info_array_t info; mach_msg_type_number_t count;
    if (host_processor_info(mach_host_self(), PROCESSOR_CPU_LOAD_INFO, &ncpu, &info, &count) != KERN_SUCCESS) return -1;
    double used = 0, total = 0;
    if (prevCpu) {
        for (natural_t i = 0; i < ncpu; i++) {
            integer_t *c = &info[i * CPU_STATE_MAX], *p = &prevCpu[i * CPU_STATE_MAX];
            double u = (c[CPU_STATE_USER] - p[CPU_STATE_USER]) + (c[CPU_STATE_SYSTEM] - p[CPU_STATE_SYSTEM]) + (c[CPU_STATE_NICE] - p[CPU_STATE_NICE]);
            used += u; total += u + (c[CPU_STATE_IDLE] - p[CPU_STATE_IDLE]);
        }
        vm_deallocate(mach_task_self(), (vm_address_t)prevCpu, prevCpuCount * sizeof(integer_t));
    }
    prevCpu = info; prevCpuCount = count;
    return total > 0 ? used / total * 100.0 : 0;
}

static double ramUsedGB(void) {
    vm_statistics64_data_t vm; mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm, &cnt) != KERN_SUCCESS) return -1;
    vm_size_t page; host_page_size(mach_host_self(), &page);
    uint64_t used = ((uint64_t)vm.internal_page_count - vm.purgeable_count + vm.wire_count + vm.compressor_page_count) * page;
    return used / 1073741824.0;
}

int main(void) {
    @autoreleasepool {
        uint64_t memsize = 0; size_t len = sizeof(memsize);
        sysctlbyname("hw.memsize", &memsize, &len, NULL, 0);

        // Công suất
        CFMutableDictionaryRef chans = IOReportCopyChannelsInGroup(CFSTR("Energy Model"), NULL, 0, 0, 0);
        CFMutableDictionaryRef subbed = NULL;
        IOReportSubscriptionRef sub = chans ? IOReportCreateSubscription(NULL, chans, &subbed, 0, NULL) : NULL;
        CFDictionaryRef prevSample = sub ? IOReportCreateSamples(sub, subbed, NULL) : NULL;

        smcInit();

        cpuUsage();
        NSDate *prevTime = [NSDate date];
        setvbuf(stdout, NULL, _IOLBF, 0);

        while (1) {
            @autoreleasepool {
                usleep(500000);
                double dt = -[prevTime timeIntervalSinceNow];
                prevTime = [NSDate date];

                double cpuW = 0, gpuW = 0, aneW = 0; BOOL havePower = NO;
                if (sub) {
                    CFDictionaryRef cur = IOReportCreateSamples(sub, subbed, NULL);
                    CFDictionaryRef delta = IOReportCreateSamplesDelta(prevSample, cur, NULL);
                    CFRelease(prevSample); prevSample = cur;
                    NSArray *items = ((__bridge NSDictionary *)delta)[@"IOReportChannels"];
                    for (id item in items) {
                        CFDictionaryRef ch = (__bridge CFDictionaryRef)item;
                        NSString *name = (__bridge NSString *)IOReportChannelGetChannelName(ch);
                        NSString *unit = (__bridge NSString *)IOReportChannelGetUnitLabel(ch);
                        if (!name) continue;
                        double w = energyToJoules(IOReportSimpleGetIntegerValue(ch, 0), [unit stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet]) / dt;
                        if ([name hasSuffix:@"CPU Energy"]) { cpuW += w; }
                        else if ([name isEqualToString:@"GPU Energy"]) { gpuW += w; havePower = YES; }
                        else if ([name hasPrefix:@"ANE"]) { aneW += w; havePower = YES; }
                    }
                    CFRelease(delta);
                }

                double cpuTemp = smcConn ? smcAvg(cpuTempKeys, nCpuTemp) : -1;
                double gpuTemp = smcConn ? smcAvg(gpuTempKeys, nGpuTemp) : -1;
                double sysW = sysPowerKey ? smcReadFloat(sysPowerKey, 4) : -1;
                // Trên M5/macOS 27, IOReport không báo năng lượng CPU nếu không có root -> cpu_w = -1, dùng soc_w.
                if (cpuW > 1.0) haveCpu = YES;  // bỏ qua giá trị lẻ tẻ gần 0
                if (!haveCpu) cpuW = -1;
                double socW = socPowerKey ? smcReadFloat(socPowerKey, 4) : -1;

                printf("{\"cpu\":%.1f,\"gpu\":%.1f,\"ram\":%.2f,\"ram_total\":%.0f,"
                       "\"cpu_w\":%.2f,\"gpu_w\":%.2f,\"ane_w\":%.2f,\"soc_w\":%.1f,\"power\":%s,"
                       "\"gpu_temp\":%.1f,\"cpu_temp\":%.1f,\"sys_w\":%.1f}\n",
                       cpuUsage(), gpuUtilization(), ramUsedGB(), memsize / 1073741824.0,
                       cpuW, gpuW, aneW, socW, havePower ? "true" : "false", gpuTemp, cpuTemp, sysW);
            }
        }
    }
}
