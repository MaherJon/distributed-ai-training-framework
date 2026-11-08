pluginManagement {
    repositories {
        // 国内网络优先走镜像；官方仓库作为兜底
        maven("https://maven.aliyun.com/repository/gradle-plugin")
        maven("https://maven.aliyun.com/repository/public")
        gradlePluginPortal()
        mavenCentral()
        google()
    }
}
rootProject.name = "mahe-ui"
