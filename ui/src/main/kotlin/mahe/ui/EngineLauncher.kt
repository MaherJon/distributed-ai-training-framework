package mahe.ui

import java.io.File
import java.net.InetSocketAddress
import java.net.Socket

/**
 * 一体化安装包里的 Python 引擎启动器。
 *
 * `build.py --bundle` 会把 PyInstaller 冻结出的 mahe-engine 放进安装包的
 * resources/engine/ 下；UI 启动时若 47830 端口无人监听，就拉起引擎
 * （默认 --role demo，可用环境变量 MAHE_ENGINE_ROLE 覆盖为 master/worker）。
 * 开发模式（gradlew run）下找不到引擎时静默跳过，保持原有手动启动流程。
 */
object EngineLauncher {
    private const val HOST = "127.0.0.1"
    private const val PORT = 47830

    private var process: Process? = null

    fun ensureRunning() {
        if (process?.isAlive == true || isPortOpen()) return
        val exe = engineExecutable() ?: return
        val logFile = File(System.getProperty("user.home"), ".mahe")
            .apply { mkdirs() }
            .resolve("engine.log")
        process = try {
            ProcessBuilder(exe.absolutePath, "--role", role())
                .redirectErrorStream(true)
                .redirectOutput(ProcessBuilder.Redirect.appendTo(logFile))
                .start()
        } catch (_: Exception) {
            null
        }
    }

    fun stop() {
        process?.let { p ->
            p.descendants().forEach { it.destroy() }
            p.destroy()
        }
        process = null
    }

    private fun role(): String =
        System.getenv("MAHE_ENGINE_ROLE")?.lowercase()
            ?.takeIf { it in setOf("demo", "master", "worker") } ?: "demo"

    private fun engineExecutable(): File? {
        val name = if (System.getProperty("os.name").lowercase().contains("win"))
            "mahe-engine.exe" else "mahe-engine"
        val roots = buildList {
            System.getProperty("compose.application.resources.dir")?.let { add(File(it)) }
            System.getProperty("jpackage.app-path")?.let {
                add(File(it).parentFile.resolve("resources"))
            }
        }
        for (root in roots) {
            val f = root.resolve("engine").resolve(name)
            if (f.isFile) return f
        }
        // 开发模式回退：build.py --bundle 把引擎放在
        // ui/resources/<os>-<arch>/engine（Compose 只打包 <os>/<os>-<arch> 子目录）
        val osName = System.getProperty("os.name").lowercase()
        val osDir = when {
            osName.contains("win") -> "windows"
            osName.contains("mac") -> "macos"
            else -> "linux"
        }
        val archDir = System.getProperty("os.arch").lowercase()
            .let { if (it == "aarch64" || it == "arm64") "arm64" else "x64" }
        val dev = File("resources/$osDir-$archDir/engine").resolve(name)
        return dev.takeIf { it.isFile }
    }

    private fun isPortOpen(): Boolean = try {
        Socket().use { it.connect(InetSocketAddress(HOST, PORT), 500) }
        true
    } catch (_: Exception) {
        false
    }
}
