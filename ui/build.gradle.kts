import org.jetbrains.compose.desktop.application.dsl.TargetFormat

plugins {
    // 单平台 JVM：源码位于 src/main/kotlin（multiplatform 的 jvmMain 不会编译这里的文件）
    kotlin("jvm") version "1.9.23"
    id("org.jetbrains.compose") version "1.6.2"
}

group = "org.mahe"
version = "0.1.0"

repositories {
    // 国内网络优先走镜像；官方仓库作为兜底
    maven("https://maven.aliyun.com/repository/public")
    maven("https://maven.aliyun.com/repository/google")
    mavenCentral()
    google()
}

kotlin {
    jvmToolchain(17)
}

dependencies {
    implementation(compose.desktop.currentOs)
    implementation(compose.material3)
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.8.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-swing:1.8.0")
}

compose.desktop {
    application {
        mainClass = "mahe.ui.MainKt"
        nativeDistributions {
            targetFormats(TargetFormat.Dmg, TargetFormat.Msi, TargetFormat.Deb)
            packageName = "MAHE"
            // jpackage/Dmg 要求主版本号 > 0，不能用 0.x
            packageVersion = "1.0.0"
            // 打进安装包的资源目录：Python 引擎由根目录 build.py --bundle 生成到
            // ui/resources/engine/，UI 启动时从 compose.application.resources.dir 读取
            appResourcesRoot = file("resources")
        }
    }
}
