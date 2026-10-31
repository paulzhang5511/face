// Android 日志初始化: log 输出到 logcat (桌面平台走 env_logger, 便于本机调试)
pub fn init_logging() {
    #[cfg(target_os = "android")]
    {
        use log::LevelFilter;
        android_logger::init_once(
            android_logger::Config::default()
                .with_tag("face-bench")
                .with_max_level(LevelFilter::Info),
        );
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = env_logger::builder()
            .filter_level(log::LevelFilter::Info)
            .try_init();
    }
}
